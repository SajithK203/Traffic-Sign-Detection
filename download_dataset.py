import os
import sys
import time
import urllib.request
import urllib.error

url = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip"
file_path = r"C:\Users\sigun\Desktop\Traffic-Sign-Detection\Traffic-Sign-Detection\data\raw\gtsdb\TrainIJCNN2013.zip"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

max_retries = 100
retry_delay = 3

total_size = 1693630656 # approx 1.6 GB

print(f"Starting resilient download for {url}")

for attempt in range(max_retries):
    downloaded_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    if total_size and downloaded_size >= total_size:
        print("Download size looks complete!")
        # Can't be exactly sure without HEAD, but we'll try to extract it later.
        break

    print(f"Attempt {attempt + 1}: Resuming from {downloaded_size / (1024*1024):.2f} MB")
    
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={downloaded_size}-")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(file_path, "ab") as f:
                last_print = time.time()
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if time.time() - last_print > 10:
                        print(f"Progress: {downloaded_size / (1024 * 1024):.2f} MB")
                        last_print = time.time()
                        
    except urllib.error.HTTPError as e:
        if e.code == 416:
            print("Received 416 Range Not Satisfiable - assuming download complete.")
            break
        print(f"HTTP Error: {e.code} {e.reason}")
        time.sleep(retry_delay)
    except Exception as e:
        print(f"Error during download: {e}")
        time.sleep(retry_delay)

print("Download script finished.")
