import pickle

import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import sklearn.metrics
from ember.features import PEFeatureExtractor
import csv


def find_in_dataset(dir, example):
    import pandas as pd
    ds = pd.read_csv(dir + 'samples.csv')
    # keys=ds.keys().to_numpy()
    ds = ds.to_dict('records')
    for i in range(len(ds)):
        file_data = open(dir + "samples/" + str(ds[i]["id"]), "rb").read()
        extractor = PEFeatureExtractor(2)
        pe_extracted = np.array(extractor.feature_vector(file_data), dtype=np.float64)
        if np.array(example) == pe_extracted:
            print("ok")
    print("No")
