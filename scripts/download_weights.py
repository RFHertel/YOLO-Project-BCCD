#download_weights.py

import urllib.request
import os

def download_with_progress(url, filepath):
    def download_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"Downloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
    
    print(f"Downloading darknet53 weights...")
    urllib.request.urlretrieve(url, filepath, reporthook=download_hook)
    print("\nDownload complete!")
    
    # Verify
    size = os.path.getsize(filepath)
    print(f"File size: {size/1024/1024:.2f} MB")
    
    if size < 160000000:  # Should be ~162MB
        print("❌ Download failed! File too small.")
        return False
    else:
        print("✅ Download successful!")
        return True

# Download the weights
url = "https://pjreddie.com/media/files/darknet53_448.weights"
output = "weight/darknet53_448.weights"

# Remove old file if exists
if os.path.exists(output):
    os.remove(output)

# Download
success = download_with_progress(url, output)

if not success:
    print("\nTrying alternative download method...")
    import requests
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100
                print(f"Progress: {percent:.1f}%", end='\r')
    
    print(f"\nFinal size: {os.path.getsize(output)/1024/1024:.2f} MB")