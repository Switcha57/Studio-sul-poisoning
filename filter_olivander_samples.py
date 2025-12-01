import json
import os
import shutil
import sys

def main():
    # Paths
    base_dir = os.getcwd()
    source_dir = os.path.join(base_dir, 'PoisonedSamples', 'Olivander')
    dest_dir = os.path.join(base_dir, 'PoisonedSamples', 'Olivander_VT')
    vt_results_path = os.path.join(base_dir, 'vt_analysis_results.json')

    # 1. Copy directory
    if os.path.exists(dest_dir):
        print(f"Destination directory {dest_dir} already exists. Removing it first.")
        shutil.rmtree(dest_dir)
    
    print(f"Copying {source_dir} to {dest_dir}...")
    shutil.copytree(source_dir, dest_dir)

    # 2. Load VT results
    print(f"Loading {vt_results_path}...")
    try:
        with open(vt_results_path, 'r') as f:
            vt_data = json.load(f)
    except FileNotFoundError:
        print("Error: vt_analysis_results.json not found.")
        sys.exit(1)

    # 3. Build a lookup map
    # Map: (subdir, core_name) -> malicious_count
    lookup_map = {}
    
    for key, value in vt_data.items():
        # /workspace/file/Olivander/dnn_kdde4000/final-18-step-1000-section-1-adv.exe
        parts = key.split('/')
        
        try:
            olivander_index = parts.index('Olivander')
            subdir = parts[olivander_index + 1]
            filename = parts[-1]
            
            if filename.startswith('final-') and filename.endswith('.exe'):
                core_name = filename[6:-4] # Remove 'final-' and '.exe'
                
                stats = value.get('stats', {})
                malicious_count = stats.get('malicious', 0)
                
                lookup_map[(subdir, core_name)] = malicious_count
        except (ValueError, IndexError):
            # Path structure might be different, skip or log
            continue

    print(f"Built lookup map with {len(lookup_map)} entries.")

    # 4. Iterate and filter files
    removed_count = 0
    kept_count = 0
    
    for root, dirs, files in os.walk(dest_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
                
            # Get subdir relative to dest_dir
            rel_path = os.path.relpath(root, dest_dir)

            
            subdir = rel_path
            if subdir == '.':
                continue

            core_name = file[:-5] # Remove .json
            
            # Check lookup
            key = (subdir, core_name)
            
            if key in lookup_map:
                malicious_count = lookup_map[key]
                if malicious_count == 0:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    
                    # Also remove the associated .exe file
                    exe_filename = f"final-{core_name}.exe"
                    exe_path = os.path.join(root, exe_filename)
                    if os.path.exists(exe_path):
                        os.remove(exe_path)
                        # print(f"Removed {exe_path}")
                    
                    removed_count += 1
                    # print(f"Removed {file_path} (malicious: 0)")
                else:
                    kept_count += 1
            else:
                # File not found in VT results
                # print(f"Warning: {file} in {subdir} not found in VT results. Keeping.")
                kept_count += 1

    print(f"Finished. Removed {removed_count} files. Kept {kept_count} files.")

if __name__ == "__main__":
    main()
