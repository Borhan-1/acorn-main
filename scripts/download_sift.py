import urllib.request
import os
import tarfile
import sys

def download_and_extract_sift1m(target_dir="./Datasets/sift1M"):
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "sift.tar.gz")
    
    # Check if already extracted
    expected_files = ["sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"]
    if all(os.path.exists(os.path.join(target_dir, f)) for f in expected_files):
        print(f"SIFT1M dataset is already downloaded and extracted in {target_dir}")
        return True
    
    # Potential mirror URLs
    urls = [
        "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "https://dl.fbaipublicfiles.com/faiss/sift1M.tar.gz",
    ]
    
    downloaded = False
    for url in urls:
        print(f"Attempting download from: {url} ...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response, open(archive_path, 'wb') as out_file:
                total_size = int(response.getheader('Content-Length', 0))
                downloaded_size = 0
                block_size = 1024 * 1024  # 1MB
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded_size += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = downloaded_size * 100 / total_size
                        sys.stdout.write(f"\rDownloading: {downloaded_size/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({percent:.1f}%)")
                    else:
                        sys.stdout.write(f"\rDownloading: {downloaded_size/(1024*1024):.1f}MB")
                    sys.stdout.flush()
                print("\nDownload completed!")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except Exception:
                    pass
                    
    if not downloaded:
        print("Could not download automatically. Please check network connection.")
        return False
        
    print(f"Extracting {archive_path} to {target_dir} ...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Extract contents directly into target_dir
            for member in tar.getmembers():
                filename = os.path.basename(member.name)
                if filename:
                    source = tar.extractfile(member)
                    target = os.path.join(target_dir, filename)
                    with open(target, "wb") as f_out:
                        f_out.write(source.read())
                    print(f"Extracted: {filename}")
        print("Dataset ready!")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

if __name__ == "__main__":
    download_and_extract_sift1m()

