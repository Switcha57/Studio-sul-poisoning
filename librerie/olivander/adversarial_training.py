import argparse
import ast
import configparser

import numpy as np
import tensorflow as tf

import pickle
from libs.adv_train import adv_scores
from libs.build_dnn import build_dnn
from libs.dataset_to_features import dataset_to_features

tf.compat.v1.disable_eager_execution()

num_thread = 1
OPTIMIZED = False
config = configparser.ConfigParser()
config.read('config.ini')

pe_folder = config.get("SETTINGS", "pe_folder")
counterfactual_path = config.get("SETTINGS", "counterfactual_path")

model = build_dnn()
model.load_weights("models/DNN_w.h5")
eps_array = config.get("ADVERSARIAL_TRAINING", "EPS_ARRAY")
eps_array=ast.literal_eval(eps_array)
print("using EPS array:"+str(eps_array))
to_test = {}

# LOAD EXTRACTED LIEF DATASET
with open(config.get("SETTINGS", "lief_dataset"), 'rb') as handle:
    ds = pickle.load(handle)
x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
del ds

res_i = []
with open(counterfactual_path, "rb") as h:
    data = pickle.load(h)
my_100 = []
for i in data.keys():
    found, not_found, differences, differences_index, target, test, times = data[i]
    my_100.append(test[0][0][0:-1])
scores_tot = []
for ind in eps_array:
    eps = ind
    model2 = build_dnn()
    model2.load_weights("models/DNN_w.h5")
    _, scores = adv_scores(model2, x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta,
                           x_test_meta, eps, np.array(my_100), np.array([1 for i in range(0, len(my_100))]))
    scores_tot.append(scores)
    new_d = []
