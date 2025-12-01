#!/usr/bin/env python3
"""
VirusTotal File Upload Script

This script uploads files to VirusTotal using multiple API keys with rate limiting.
Handles files larger than 32MB and tracks progress.

Free API Limits:
- 4 requests per minute per API key
- Files up to 32MB: direct upload
- Files larger than 32MB: special upload URL required
"""

import os
import sys
import time
import json
import hashlib
import requests
from datetime import datetime
import vt


class RateLimitedAPI:
    """Manages API key rotation and rate limiting for VirusTotal free tier."""
    
    def __init__(self, api_keys):
        """
        Initialize with multiple API keys.
        
        Args:
            api_keys: List of VirusTotal API keys
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.request_times = {key: [] for key in api_keys}
        self.requests_per_minute = 4
        self.minute_window = 60  # seconds
        
    def get_client(self):
        """
        Get a VirusTotal client with an available API key.
        Blocks if all keys are rate limited.
        
        Returns:
            vt.Client instance
        """
        while True:
            # Try each key
            for i in range(len(self.api_keys)):
                key_index = (self.current_key_index + i) % len(self.api_keys)
                api_key = self.api_keys[key_index]
                
                # Clean up old request times
                current_time = time.time()
                self.request_times[api_key] = [
                    t for t in self.request_times[api_key]
                    if current_time - t < self.minute_window
                ]
                
                # Check if we can use this key
                if len(self.request_times[api_key]) < self.requests_per_minute:
                    self.current_key_index = key_index
                    self.request_times[api_key].append(current_time)
                    return vt.Client(api_key)
            
            # All keys are rate limited, wait
            wait_time = self._calculate_wait_time()
            print(f"⏳ All API keys rate limited. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    
    def _calculate_wait_time(self):
        """Calculate how long to wait until a key becomes available."""
        current_time = time.time()
        min_wait = float('inf')
        
        for api_key in self.api_keys:
            if self.request_times[api_key]:
                oldest_request = min(self.request_times[api_key])
                wait_time = self.minute_window - (current_time - oldest_request) + 1
                min_wait = min(min_wait, wait_time)
        
        return max(min_wait, 1)


class VirusTotalUploader:
    """Handles file uploads to VirusTotal with progress tracking."""
    
    def __init__(self, api_keys, progress_file='vt_upload_progress.json'):
        """
        Initialize the uploader.
        
        Args:
            api_keys: List of VirusTotal API keys
            progress_file: Path to save upload progress
        """
        self.rate_limiter = RateLimitedAPI(api_keys)
        self.progress_file = progress_file
        self.progress = self._load_progress()
        self.large_file_threshold = 32 * 1024 * 1024  # 32 MB in bytes
        
    def _load_progress(self):
        """Load progress from previous runs."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load progress file: {e}")
        return {'uploaded': {}, 'failed': {}}
    
    def _save_progress(self):
        """Save current progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save progress: {e}")
    
    def _get_file_hash(self, file_path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _should_skip_file(self, file_path):
        """
        Check if file should be skipped based on extension or name.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file should be skipped
        """
        file_name = os.path.basename(file_path)
        
        # Skip files starting with 'temp'
        if file_name.lower().startswith('temp'):
            return True
        
        # Skip certain extensions
        skip_extensions = {'.json', '.pickle', '.pkl', '.txt'}
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in skip_extensions:
            return True
        
        return False
    
    def upload_file(self, file_path):
        """
        Upload a single file to VirusTotal.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            tuple: (success, result_or_error)
        """
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        try:
            # Get a client (handles rate limiting)
            client = self.rate_limiter.get_client()
            
            if False and file_size > self.large_file_threshold:
                # Large file: get special upload URL and upload directly
                print(f"  📤 Large file ({file_size / (1024*1024):.2f} MB), requesting upload URL...")
                
                # Get the special upload URL for large files
                upload_url_response = client.get_json("/files/upload_url")
                upload_url = upload_url_response["data"]
                
                # Upload file directly to the special URL using requests
                with open(file_path, 'rb') as f:
                    files = {'file': (file_name, f)}
                    # Timeout set to 300 seconds (5 minutes) for large file uploads
                    response = requests.post(upload_url, files=files, timeout=300)
                    response.raise_for_status()
                    upload_result = response.json()
                
                client.close()
                
                # Extract analysis information from response
                analysis_id = upload_result['data']['id']
                result = {
                    'analysis_id': analysis_id,
                    'file_size': file_size,
                    'uploaded_at': datetime.now().isoformat(),
                    'sha256': self._get_file_hash(file_path),
                    'type': 'large_file'
                }
            else:
                # Normal file: direct upload using vt library
                with open(file_path, 'rb') as f:
                    analysis = client.scan_file(f)
                
                client.close()
                
                result = {
                    'analysis_id': analysis.id,
                    'file_size': file_size,
                    'uploaded_at': datetime.now().isoformat(),
                    'sha256': self._get_file_hash(file_path),
                    'type': 'normal'
                }
            
            return True, result
            
        except vt.error.APIError as e:
            error_msg = f"API Error: {e}"
            print(f"  ❌ {error_msg}")
            return False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Upload Error: {e}"
            print(f"  ❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"  ❌ {error_msg}")
            return False, error_msg
    
    def scan_directory(self, directory):
        """
        Scan directory and upload all eligible files.
        
        Args:
            directory: Root directory to scan
        """
        print(f"🔍 Scanning directory: {directory}")
        
        # Collect all files
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if not self._should_skip_file(file_path):
                    all_files.append(file_path)
        
        print(f"📁 Found {len(all_files)} files to upload")
        
        # Filter out already uploaded files
        files_to_upload = [
            f for f in all_files 
            if f not in self.progress['uploaded']
        ]
        
        if len(files_to_upload) < len(all_files):
            print(f"✅ {len(all_files) - len(files_to_upload)} files already uploaded")
        
        print(f"📤 Uploading {len(files_to_upload)} files...")
        print(f"⏰ Estimated time: ~{len(files_to_upload) * 15 / 60:.1f} minutes\n")
        
        # Upload files
        for i, file_path in enumerate(files_to_upload, 1):
            relative_path = os.path.relpath(file_path, directory)
            file_size = os.path.getsize(file_path)
            
            print(f"[{i}/{len(files_to_upload)}] {relative_path}")
            print(f"  Size: {file_size / (1024*1024):.2f} MB")
            
            success, result = self.upload_file(file_path)
            
            if success:
                print(f"  ✅ Uploaded successfully!")
                print(f"  Analysis ID: {result['analysis_id']}")
                self.progress['uploaded'][file_path] = result
            else:
                print(f"  ❌ Failed: {result}")
                self.progress['failed'][file_path] = {
                    'error': result,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Save progress after each file
            self._save_progress()
            print()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Upload Summary")
        print("="*60)
        print(f"✅ Successfully uploaded: {len(self.progress['uploaded'])} files")
        print(f"❌ Failed: {len(self.progress['failed'])} files")
        
        if self.progress['failed']:
            print("\nFailed files:")
            for file_path, info in self.progress['failed'].items():
                print(f"  - {os.path.basename(file_path)}: {info['error']}")


def main():
    """Main entry point for the script."""
    # VirusTotal API Keys (replace with your actual keys)
    API_KEYS = [

    ]
    
    # Validate API keys
    if any(key.startswith('YOUR_API_KEY') for key in API_KEYS):
        print("❌ Error: Please replace the placeholder API keys with your actual VirusTotal API keys")
        print("Edit the API_KEYS list in vt_scanner.py")
        sys.exit(1)
    
    # Directory to scan
    UPLOAD_DIRECTORY = '/workspace/file'
    
    if not os.path.exists(UPLOAD_DIRECTORY):
        print(f"❌ Error: Directory not found: {UPLOAD_DIRECTORY}")
        sys.exit(1)
    
    print("="*60)
    print("🛡️  VirusTotal File Upload Script")
    print("="*60)
    print(f"📁 Upload directory: {UPLOAD_DIRECTORY}")
    print(f"🔑 Using {len(API_KEYS)} API keys")
    print(f"⚙️  Rate limit: 4 requests/minute per key")
    print(f"📋 Progress file: vt_upload_progress.json")
    print("="*60 + "\n")
    
    # Create uploader and start scanning
    uploader = VirusTotalUploader(API_KEYS)
    
    try:
        uploader.scan_directory(UPLOAD_DIRECTORY)
    except KeyboardInterrupt:
        print("\n\n⚠️  Upload interrupted by user")
        print("Progress has been saved. Run the script again to resume.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✨ Upload process completed!")


if __name__ == '__main__':
    main()
