import json
import os
import struct
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq


def download_file(url: str, dst: str, max_retries: int = 8) -> None:
    for attempt in range(1, max_retries + 1):
        existing = os.path.getsize(dst) if os.path.exists(dst) else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                status = getattr(resp, "status", 200)
                if status == 206:
                    mode = "ab"
                    done = existing
                    cl = int(resp.getheader("Content-Length", 0))
                    total = existing + cl if cl > 0 else 0
                else:
                    mode = "wb"
                    done = 0
                    total = int(resp.getheader("Content-Length", 0))
                with open(dst, mode) as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total > 0:
                            pct = done * 100.0 / total
                            sys.stdout.write(
                                f"\r  {os.path.basename(dst)}: {done / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({pct:.1f}%)"
                            )
                        else:
                            sys.stdout.write(f"\r  {os.path.basename(dst)}: {done / (1024*1024):.1f} MB")
                        sys.stdout.flush()
            print()
            if total == 0 or done >= total:
                return
        except urllib.error.HTTPError as e:
            if e.code == 416:
                print(f"\n  {os.path.basename(dst)}: already complete.")
                return
            print(f"\n  Attempt {attempt}/{max_retries} HTTP {e.code}, retrying...")
        except Exception as e:
            print(f"\n  Attempt {attempt}/{max_retries} error ({e}), resuming from {os.path.getsize(dst) if os.path.exists(dst) else 0} bytes...")
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def prepare_dbpedia_openai(
    target_dir: str = "./Datasets/dbpedia_openai100k",
    target_base: int = 100000,
    target_queries: int = 1000,
) -> bool:
    os.makedirs(target_dir, exist_ok=True)
    base_path = os.path.join(target_dir, "sift_base.fvecs")
    query_path = os.path.join(target_dir, "sift_query.fvecs")
    meta_path = os.path.join(target_dir, "word_lengths.ivecs")

    if (
        os.path.exists(base_path)
        and os.path.getsize(base_path) > 1024 * 1024
        and os.path.exists(query_path)
        and os.path.getsize(query_path) > 1024
        and os.path.exists(meta_path)
    ):
        print(f"DBpedia-OpenAI-1536d dataset already prepared in {target_dir}")
        return True

    tree_url = "https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M/tree/main/data"
    print(f"Fetching shard list from: {tree_url}")
    req = urllib.request.Request(tree_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        entries = json.loads(resp.read().decode("utf-8"))

    parquet_paths = sorted(
        e["path"] for e in entries if e.get("path", "").endswith(".parquet")
    )
    if not parquet_paths:
        print("ERROR: No parquet shards found.")
        return False

    total_needed = target_base + target_queries
    base_count = 0
    query_count = 0
    global_idx = 0
    stride_q = max(1, total_needed // target_queries)
    dim = 1536

    with open(base_path, "wb") as f_base, open(query_path, "wb") as f_query, open(meta_path, "wb") as f_meta:
        for s_idx, rel_path in enumerate(parquet_paths):
            if base_count >= target_base and query_count >= target_queries:
                break
            url = f"https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/{rel_path}"
            local_pq = os.path.join(target_dir, f"shard_{s_idx}.parquet")
            print(f"Downloading/resuming shard {s_idx + 1} ({rel_path})...")
            download_file(url, local_pq)

            pf = pq.ParquetFile(local_pq)
            cols = pf.schema_arrow.names
            emb_col = "openai" if "openai" in cols else ("emb" if "emb" in cols else cols[-1])
            text_col = "text" if "text" in cols else ("title" if "title" in cols else None)
            read_cols = [emb_col] + ([text_col] if text_col else [])
            print(
                f"Converting shard {s_idx} (rows={pf.metadata.num_rows}, "
                f"cols={cols}, emb={emb_col}, meta={text_col})..."
            )

            for batch in pf.iter_batches(batch_size=4096, columns=read_cols):
                col0 = batch.column(0)
                if hasattr(col0, "values") and len(col0.values) == len(col0) * dim:
                    embs = col0.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, dim)
                else:
                    embs = np.asarray(col0.to_pylist(), dtype=np.float32)
                if embs.ndim != 2 or embs.shape[1] != dim:
                    continue
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embs = embs / norms

                if text_col:
                    texts = batch.column(1).to_pylist()
                    metas = [len(t) if isinstance(t, str) else 0 for t in texts]
                else:
                    metas = [0] * len(embs)

                n_rows = embs.shape[0]
                dim_col = np.full((n_rows, 1), dim, dtype=np.int32)
                packed = np.hstack([dim_col.view(np.float32), embs])

                for r in range(n_rows):
                    if base_count >= target_base and query_count >= target_queries:
                        break
                    is_q = (query_count < target_queries) and (global_idx % stride_q == 0)
                    global_idx += 1
                    if is_q:
                        f_query.write(packed[r].tobytes())
                        query_count += 1
                    elif base_count < target_base:
                        f_base.write(packed[r].tobytes())
                        f_meta.write(struct.pack("<ii", 1, metas[r]))
                        base_count += 1

            try:
                os.remove(local_pq)
                print(f"Removed temporary {os.path.basename(local_pq)}")
            except OSError:
                pass

    print(
        f"Prepared DBpedia-OpenAI-1536d (d={dim}): {base_count} base vectors, "
        f"{query_count} queries -> {target_dir}"
    )
    return (base_count == target_base) and (query_count == target_queries)


if __name__ == "__main__":
    ok = prepare_dbpedia_openai()
    sys.exit(0 if ok else 1)
