import numpy as np
from ember.features import PEFeatureExtractor
import pandas as pd


def create_dataset(pe_dir, save=True, output_file='pickle/dataset_lief.pickle'):

    ds = pd.read_csv(pe_dir + 'samples.csv')
    ds = ds.to_dict('records')
    print("Calculating lief encoding on a dataset of "+str(len(ds))+" PE files")
    for i in range(len(ds)):
        file_data = open(pe_dir + "samples/" + str(ds[i]["id"]), "rb").read()
        extractor = PEFeatureExtractor(2)
        pe_extracted = np.array(extractor.feature_vector(file_data), dtype=np.float64)
        ds[i]["features"] = pe_extracted
        print(str(i)+"/"+str(len(ds)))
    import pickle
    if save:
        with open(output_file, 'wb') as handle:
            pickle.dump(ds, handle)
    return ds
