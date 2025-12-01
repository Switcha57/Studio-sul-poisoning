import os
import shutil
import pandas as pd

# Configuration
CLEAN_DATASET_PATH = "cleanDataset/KDDE_samples/train"
POISONED_SAMPLES_BASE = "PoisonedSamples"
OUTPUT_BASE = "poisonedDataset/NoRimpiazzo/clean"

# Poison types and their subpaths (assuming dnn_kdde4000 as per context)
POISON_SOURCES = [
    ("GammaMalConv", "Gamma/malconv"),
    ("GammaEmber", "Gamma/dnn_ember2018"),
    ("Olivander", "Olivander/dnn_ember2018")
    # ("Olivander_VT", "Olivander_VT/dnn_kdde4000")
]

def create_dataset(poison_name, poison_subpath):
    print(f"Processing {poison_name}...")
    
    # Define destination path
    # e.g. poisonedDataset/NoRimpiazzo/clean/Gamma_dnn_kdde4000
    dest_path = os.path.join(OUTPUT_BASE, f"{poison_name}")
    
    # Remove if exists to start fresh
    if os.path.exists(dest_path):
        print(f"Removing existing directory: {dest_path}")
        shutil.rmtree(dest_path)
        
    # Copy clean dataset
    print(f"Copying clean dataset from {CLEAN_DATASET_PATH} to {dest_path}...")
    shutil.copytree(CLEAN_DATASET_PATH, dest_path)
    
    # Path to poisoned samples
    poison_source_path = os.path.join(POISONED_SAMPLES_BASE, poison_subpath)
    
    if not os.path.exists(poison_source_path):
        print(f"Warning: {poison_source_path} does not exist. Skipping.")
        return

    # Prepare to update CSV
    goodware_csv_path = os.path.join(dest_path, "goodware", "samples.csv")
    
    count = 0
    new_rows = []
    
    print(f"Adding poisoned samples from {poison_source_path}...")
    
    for file_name in os.listdir(poison_source_path):
        src_file = os.path.join(poison_source_path, file_name)
        if not os.path.isfile(src_file):
            continue
            
        # Append "goody" to the file name as requested
        new_file_name = file_name + "goody"
        
        dest_file = os.path.join(dest_path, "goodware", "samples", new_file_name)
        
        # Copy the file
        shutil.copy2(src_file, dest_file)
        
        # Add to CSV data
        # Format: id,features,list
        new_rows.append({"id": new_file_name, "features": "", "list": "Whitelist"})
        count += 1
        
    print(f"Added {count} poisoned samples to goodware.")
    
    # Update CSV
    if new_rows:
        df = pd.DataFrame(new_rows)
        # Append to existing CSV
        df.to_csv(goodware_csv_path, mode='a', header=False, index=False)
        print(f"Updated {goodware_csv_path} with new samples.")
    else:
        print("No samples found to add.")

def main():
    # Ensure output base directory exists
    if not os.path.exists(OUTPUT_BASE):
        os.makedirs(OUTPUT_BASE)
        
    for name, subpath in POISON_SOURCES:
        create_dataset(name, subpath)

if __name__ == "__main__":
    main()
