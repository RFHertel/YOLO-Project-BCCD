# Run: python scripts/download_voc.py --root ./data --years 2007 2012
"""
Pascal VOC Dataset Retrieval Script - Enhanced Version with Alternative Mirrors
"""

import os
import tarfile
import urllib.request
import urllib.error
import socket
from pathlib import Path
import argparse
import hashlib
from tqdm import tqdm
import time

class VOCDownloader:
    """Enhanced Pascal VOC downloader with fallback mirrors"""
    
    # Alternative download sources (these mirrors usually work better)
    DATASETS = {
        'VOC2007': {
            'trainval': {
                'url': 'http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar',
                'fallback': 'https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar',
                'size': 460032000,
                'md5': 'c52e279531787c972589f7e41ab4ae64'
            },
            'test': {
                'url': 'http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar', 
                'fallback': 'https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar',
                'size': 438030000,
                'md5': 'b6e924de25625d8de591ea690078ad9f'
            }
        },
        'VOC2012': {
            'trainval': {
                'url': 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',
                'fallback': 'https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',
                'size': 1999000000,
                'md5': '6cd6e144f989b92b3379bac3b3de84fd'
            }
        }
    }
    
    def __init__(self, root_dir='./data', timeout=60):
        """Initialize downloader with configuration"""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        socket.setdefaulttimeout(timeout)
        self.successful_downloads = []
        self.failed_downloads = []
    
    def download_file(self, url, destination, expected_size=None, desc=None):
        """Download file with progress bar"""
        
        # Check if file exists and is valid
        if destination.exists():
            file_size = destination.stat().st_size
            if expected_size and abs(file_size - expected_size) < expected_size * 0.1:  # 10% tolerance
                print(f"✓ {destination.name} already exists ({file_size / 1e9:.1f} GB)")
                return True
            elif not expected_size and file_size > 0:
                print(f"✓ {destination.name} already exists")
                return True
        
        print(f"⬇ Downloading {desc or destination.name}...")
        print(f"  URL: {url}")
        
        try:
            # Try download with progress bar
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as pbar:
                def download_hook(block_num, block_size, total_size):
                    if pbar.total != total_size:
                        pbar.total = total_size
                    downloaded = block_num * block_size
                    pbar.update(downloaded - pbar.n)
                
                urllib.request.urlretrieve(url, destination, reporthook=download_hook)
            
            # Verify size
            actual_size = destination.stat().st_size
            if expected_size and abs(actual_size - expected_size) > expected_size * 0.1:
                print(f"⚠ Size mismatch: got {actual_size / 1e9:.1f} GB, expected {expected_size / 1e9:.1f} GB")
                destination.unlink()
                return False
            
            print(f"✓ Downloaded successfully ({actual_size / 1e9:.1f} GB)")
            return True
            
        except Exception as e:
            print(f"✗ Download failed: {str(e)[:100]}")
            if destination.exists():
                destination.unlink()
            return False
    
    def extract_tar(self, tar_path, extract_to):
        """Extract tar file with progress bar"""
        print(f"📦 Extracting {tar_path.name}...")
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                with tqdm(total=len(members), desc="Extracting") as pbar:
                    for member in members:
                        tar.extract(member, extract_to)
                        pbar.update(1)
            print(f"✓ Extracted successfully")
            return True
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    def download_voc(self, year='2012', split='trainval'):
        """Download specific VOC dataset with fallback"""
        
        year_key = f'VOC{year}'
        if year_key not in self.DATASETS:
            print(f"✗ Year {year} not supported")
            return False
        
        if split not in self.DATASETS[year_key]:
            print(f"✗ Split '{split}' not available for {year_key}")
            return False
        
        dataset_info = self.DATASETS[year_key][split]
        
        # Setup paths
        download_dir = self.root_dir / 'downloads'
        download_dir.mkdir(exist_ok=True)
        
        filename = os.path.basename(dataset_info['url'])
        tar_path = download_dir / filename
        
        # Try primary URL
        success = self.download_file(
            url=dataset_info['url'],
            destination=tar_path,
            expected_size=dataset_info['size'],
            desc=f"{year_key} {split}"
        )
        
        # Try fallback URL if primary fails
        if not success and 'fallback' in dataset_info:
            print(f"🔄 Trying fallback URL...")
            success = self.download_file(
                url=dataset_info['fallback'],
                destination=tar_path,
                expected_size=dataset_info['size'],
                desc=f"{year_key} {split} (fallback)"
            )
        
        if not success:
            self.failed_downloads.append(f"{year_key}-{split}")
            print(f"\n❌ Failed to download {year_key} {split}")
            print(f"💡 Manual download option:")
            print(f"   1. Download from: {dataset_info['url']}")
            print(f"   2. Save to: {tar_path.absolute()}")
            print(f"   3. Run script again - it will detect the file")
            return False
        
        # Extract the archive
        extract_path = self.root_dir / 'VOCdevkit'
        if not (extract_path / year_key).exists():
            if not self.extract_tar(tar_path, self.root_dir):
                self.failed_downloads.append(f"{year_key}-{split}")
                return False
        else:
            print(f"✓ {year_key} already extracted")
        
        self.successful_downloads.append(f"{year_key}-{split}")
        return True
    
    def download_all(self, years=['2007', '2012']):
        """Download all specified VOC datasets"""
        
        print("=" * 70)
        print("PASCAL VOC DATASET DOWNLOADER (Alternative Mirrors)")
        print("=" * 70)
        print("Using pjreddie.com mirrors for better connectivity\n")
        
        for year in years:
            year_key = f'VOC{year}'
            if year_key not in self.DATASETS:
                continue
            
            print(f"\n{'='*50}")
            print(f"Downloading VOC{year}")
            print('='*50)
            
            for split in self.DATASETS[year_key].keys():
                self.download_voc(year, split)
                print()  # Empty line between downloads
        
        # Final summary
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        
        if self.successful_downloads:
            print(f"\n✅ Successfully downloaded and extracted:")
            for item in self.successful_downloads:
                print(f"   • {item}")
            print(f"\n📁 Data location: {self.root_dir.absolute() / 'VOCdevkit'}")
        
        if self.failed_downloads:
            print(f"\n❌ Failed downloads:")
            for item in self.failed_downloads:
                print(f"   • {item}")
            print("\n💡 Solutions:")
            print("   1. Check your internet connection")
            print("   2. Try running the script again")
            print("   3. Download manually using the URLs provided above")
            print("   4. If behind a firewall, you may need to configure proxy settings")
        
        if not self.failed_downloads and self.successful_downloads:
            print("\n✨ All downloads completed successfully!")
            print("🎉 You can now run the analysis script")
        
        return len(self.failed_downloads) == 0

def main():
    parser = argparse.ArgumentParser(description='Pascal VOC Downloader (Enhanced)')
    parser.add_argument('--root', type=str, default='./data',
                       help='Root directory for dataset')
    parser.add_argument('--years', nargs='+', default=['2007', '2012'],
                       help='Years to download (2007 and/or 2012)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Download timeout in seconds')
    
    args = parser.parse_args()
    
    # Clear any incomplete downloads
    data_path = Path(args.root)
    if data_path.exists() and not any(data_path.iterdir()):
        print("📧 Cleaning empty data directory...")
    
    # Initialize downloader
    downloader = VOCDownloader(root_dir=args.root, timeout=args.timeout)
    
    # Download datasets
    success = downloader.download_all(years=args.years)
    
    if not success:
        print("\n⚠ Some downloads failed. See instructions above for manual download.")
        exit(1)

if __name__ == '__main__':
    main()