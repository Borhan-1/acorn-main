import os
import struct
import sys
import urllib.request
import zipfile


def download_and_prepare_glove(target_dir="./Datasets/glove400k", num_queries=1000):
    os.makedirs(target_dir, exist_ok=True)
    base_path = os.path.join(target_dir, "sift_base.fvecs")
    query_path = os.path.join(target_dir, "sift_query.fvecs")
    meta_path = os.path.join(target_dir, "word_lengths.ivecs")

    if os.path.exists(base_path) and os.path.exists(query_path):
        print(f"GloVe-400K dataset is already prepared in {target_dir}")
        return True

    zip_path = os.path.join(target_dir, "glove.6B.zip")
    urls = [
        "https://nlp.stanford.edu/data/glove.6B.zip",
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    ]

    if not os.path.exists(zip_path):
        downloaded = False
        for url in urls:
            print(f"Downloading GloVe-6B from: {url}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as out:
                    total = int(resp.getheader("Content-Length", 0))
                    done = 0
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total > 0:
                            pct = done * 100.0 / total
                            sys.stdout.write(
                                f"\r  {done / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({pct:.1f}%)"
                            )
                        else:
                            sys.stdout.write(f"\r  {done / (1024*1024):.1f} MB")
                        sys.stdout.flush()
                print("\nDownload complete!")
                downloaded = True
                break
            except Exception as e:
                print(f"\nFailed from {url}: {e}")
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                    except OSError:
                        pass
        if not downloaded:
            print("ERROR: Could not download glove.6B.zip from any mirror.")
            return False

    print("Converting glove.6B.100d.txt -> .fvecs (399,000 base + 1,000 queries)...")
    dim = 100
    dim_bytes = struct.pack("<i", dim)
    vec_fmt = f"<{dim}f"

    # Hold out every 400th vector as a query (1,000 queries evenly spread across Zipfian rank)
    base_count = 0
    query_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf, \
            zf.open("glove.6B.100d.txt") as txt_in, \
            open(base_path, "wb") as f_base, \
            open(query_path, "wb") as f_query, \
            open(meta_path, "wb") as f_meta:
        for idx, raw_line in enumerate(txt_in):
            parts = raw_line.decode("utf-8", errors="ignore").rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            floats = [float(x) for x in parts[1:]]
            # L2-normalize so L2 distance == cosine/angular distance (standard for GloVe!)
            norm_sq = sum(v * v for v in floats)
            if norm_sq > 1e-12:
                inv_norm = norm_sq ** -0.5
                floats = [v * inv_norm for v in floats]
            record = dim_bytes + struct.pack(vec_fmt, *floats)

            if query_count < num_queries and (idx % 400 == 0):
                f_query.write(record)
                query_count += 1
            else:
                f_base.write(record)
                f_meta.write(struct.pack("<ii", 1, len(word)))
                base_count += 1

    print(f"Prepared GloVe-400K (d={dim}): {base_count} base vectors, {query_count} queries.")
    # Clean up large zip to save disk space
    try:
        os.remove(zip_path)
        print("Removed temporary glove.6B.zip.")
    except OSError:
        pass
    return True


if __name__ == "__main__":
    ok = download_and_prepare_glove()
    sys.exit(0 if ok else 1)
