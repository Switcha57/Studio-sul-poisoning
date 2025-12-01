#!/usr/bin/env python3
"""
VirusTotal Analysis Results Retrieval Script

This script retrieves analysis results from VirusTotal using multiple API keys 
with rate limiting. It reads the upload progress file and fetches the scan 
results for all uploaded files.

Free API Limits:
- 4 requests per minute per API key
"""

import os
import sys
import time
import json
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


class VirusTotalAnalyzer:
    """Handles analysis result retrieval from VirusTotal with progress tracking."""
    
    def __init__(self, api_keys, progress_file='vt_upload_progress.json', 
                 results_file='vt_analysis_results.json',
                 log_file='vt_analysis_logs.txt'):
        """
        Initialize the analyzer.
        
        Args:
            api_keys: List of VirusTotal API keys
            progress_file: Path to the upload progress file
            results_file: Path to save analysis results
            log_file: Path to save logs
        """
        self.rate_limiter = RateLimitedAPI(api_keys)
        self.progress_file = progress_file
        self.results_file = results_file
        self.log_file = log_file
        self.results = self._load_results()
        
    def _load_progress(self):
        """Load upload progress from file."""
        if not os.path.exists(self.progress_file):
            print(f"❌ Error: Progress file not found: {self.progress_file}")
            print("Please run vt_scanner.py first to upload files.")
            sys.exit(1)
        
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error: Could not load progress file: {e}")
            sys.exit(1)
    
    def _load_results(self):
        """Load existing results from previous runs."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load results file: {e}")
        return {}
    
    def _save_results(self):
        """Save current results to file."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=4)
        except Exception as e:
            print(f"⚠️  Warning: Could not save results: {e}")
    
    def _log_message(self, message):
        """Append a message to the log file."""
        try:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not write to log file: {e}")
    
    def get_analysis_result(self, analysis_id):
        """
        Get analysis results for a specific analysis ID.
        
        Args:
            analysis_id: The VirusTotal analysis ID
            
        Returns:
            tuple: (success, result_or_error)
        """
        client = None
        try:
            # Get a client (handles rate limiting)
            client = self.rate_limiter.get_client()
            
            # Get the analysis object
            analysis = client.get_json(f"/analyses/{analysis_id}")
            
            client.close()
            
            # Extract relevant information
            attributes = analysis.get('data').get('attributes')
            result = {
                'status': attributes.get('status'),
                'stats': attributes.get('stats'),
                'results': attributes.get('results'),
                'date': attributes.get('date'),
                'retrieved_at': datetime.now().isoformat()
            }
            
            return True, result
            
        except vt.error.APIError as e:
            error_msg = f"API Error: {e}"
            client.close()

            return False, error_msg
        except Exception as e:
            client.close()

            error_msg = f"Error: {str(e)}"
            return False, error_msg
        
    
    def analyze_all(self):
        """
        Retrieve analysis results for all uploaded files.
        """
        print("🔍 Loading upload progress...")
        progress = self._load_progress()
        
        uploaded_files = progress.get('uploaded', {})
        
        if not uploaded_files:
            print("❌ No uploaded files found in progress file.")
            print("Please run vt_scanner.py first to upload files.")
            return
        
        print(f"📁 Found {len(uploaded_files)} uploaded files")
        
        # Filter out files that already have results
        files_to_analyze = {
            path: data for path, data in uploaded_files.items()
            if path not in self.results or 'error' in self.results.get(path, {})
        }
        
        if len(files_to_analyze) < len(uploaded_files):
            print(f"✅ {len(uploaded_files) - len(files_to_analyze)} files already analyzed")
        
        print(f"📊 Analyzing {len(files_to_analyze)} files...")
        print(f"⏰ Estimated time: ~{len(files_to_analyze) * 15 / 60:.1f} minutes\n")
        
        # Retrieve results
        success_count = 0
        error_count = 0
        
        for i, (file_path, upload_data) in enumerate(files_to_analyze.items(), 1):
            analysis_id = upload_data.get('analysis_id')
            relative_path = os.path.basename(file_path)
            
            print(f"[{i}/{len(files_to_analyze)}] {relative_path}")
            print(f"  Analysis ID: {analysis_id}")
            
            if not analysis_id:
                print(f"  ⚠️  No analysis ID found, skipping...")
                self._log_message(f"Error for {file_path}: No analysis ID")
                self.results[file_path] = {
                    'error': 'No analysis ID found',
                    'timestamp': datetime.now().isoformat()
                }
                error_count += 1
                continue
            
            success, result = self.get_analysis_result(analysis_id)
            
            if success:
                status = result.get('status', 'unknown')
                stats = result.get('stats', {})
                
                print(f"  ✅ Analysis retrieved successfully!")
                print(f"  Status: {status}")
                
                if stats:
                    print(f"  Stats: {stats}")
                    # Log detection summary
                    malicious = stats.get('malicious', 0)
                    suspicious = stats.get('suspicious', 0)
                    undetected = stats.get('undetected', 0)
                    
                    if malicious > 0:
                        print(f"  🔴 {malicious} engines detected as malicious")
                    elif suspicious > 0:
                        print(f"  🟡 {suspicious} engines marked as suspicious")
                    else:
                        print(f"  🟢 {undetected} engines found nothing")
                
                self.results[file_path] = result
                self._log_message(f"Results for {file_path}: {stats}")
                success_count += 1
            else:
                print(f"  ❌ Failed: {result}")
                self.results[file_path] = {
                    'error': result,
                    'timestamp': datetime.now().isoformat()
                }
                self._log_message(f"Error for {file_path}: {result}")
                error_count += 1
            
            # Save results after each file
            self._save_results()
            print()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Analysis Summary")
        print("="*60)
        print(f"✅ Successfully retrieved: {success_count} results")
        print(f"❌ Failed: {error_count} results")
        print(f"💾 Total results in database: {len(self.results)}")
        print(f"📁 Results saved to: {self.results_file}")
        print(f"📋 Logs saved to: {self.log_file}")
        
        # Print detection statistics
        if success_count > 0:
            self._print_detection_summary()
    
    def _print_detection_summary(self):
        """Print summary of detection statistics."""
        print("\n" + "="*60)
        print("🔍 Detection Summary")
        print("="*60)
        
        malicious_files = []
        suspicious_files = []
        clean_files = []
        
        for file_path, result in self.results.items():
            if 'error' in result:
                continue
            
            stats = result.get('stats', {})
            malicious = stats.get('malicious', 0)
            suspicious = stats.get('suspicious', 0)
            
            file_name = os.path.basename(file_path)
            
            if malicious > 0:
                malicious_files.append((file_name, malicious))
            elif suspicious > 0:
                suspicious_files.append((file_name, suspicious))
            else:
                clean_files.append(file_name)
        
        print(f"🔴 Malicious detections: {len(malicious_files)} files")
        if malicious_files:
            for file_name, count in sorted(malicious_files, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {file_name}: {count} engines")
            if len(malicious_files) > 10:
                print(f"  ... and {len(malicious_files) - 10} more")
        
        print(f"\n🟡 Suspicious detections: {len(suspicious_files)} files")
        if suspicious_files:
            for file_name, count in sorted(suspicious_files, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {file_name}: {count} engines")
            if len(suspicious_files) > 5:
                print(f"  ... and {len(suspicious_files) - 5} more")
        
        print(f"\n🟢 Clean files: {len(clean_files)} files")


def main():
    """Main entry point for the script."""
    # VirusTotal API Keys (should match vt_scanner.py)
    API_KEYS = [

    ]
    
    # Validate API keys
    if any(key.startswith('YOUR_API_KEY') for key in API_KEYS):
        print("❌ Error: Please replace the placeholder API keys with your actual VirusTotal API keys")
        sys.exit(1)
    
    print("="*60)
    print("🛡️  VirusTotal Analysis Results Retrieval Script")
    print("="*60)
    print(f"🔑 Using {len(API_KEYS)} API keys")
    print(f"⚙️  Rate limit: 4 requests/minute per key")
    print(f"📋 Progress file: vt_upload_progress.json")
    print(f"💾 Results file: vt_analysis_results.json")
    print(f"📝 Log file: vt_analysis_logs.txt")
    print("="*60 + "\n")
    
    # Create analyzer and start retrieving results
    analyzer = VirusTotalAnalyzer(API_KEYS)
    
    try:
        analyzer.analyze_all()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        print("Progress has been saved. Run the script again to resume.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✨ Analysis retrieval completed!")


if __name__ == '__main__':
    main()