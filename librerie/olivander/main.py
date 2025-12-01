import argparse
import configparser
import datetime
import os
import threading
import time

import numpy as np

import pickle
from libs.adv_step_mode import adv_step_mode
from libs.build_dnn import build_dnn
from libs.create_dataset import create_dataset
from libs.dataset_to_features import dataset_to_features
from libs.find_id import find_id
from libs.generate_counterfactual import generate_counterfactual

num_thread = os.cpu_count()
OPTIMIZED = False
config = configparser.ConfigParser()
config.read('config.ini')

parser = argparse.ArgumentParser(
    prog='OLIVANDER')

parser.add_argument("--mode", choices=['load_dataset', 'load_counterfactual', "generate_dataset"], required=True)
parser.add_argument('--eta', default=1000)
parser.add_argument('--injection_type', choices=['section', 'padding'], default="section")
parser.add_argument('--section', default=1)
parser.add_argument('--iterative', default=0)
parser.add_argument('--offsetmin', default=100)
parser.add_argument('--offsetmax', default=200)
parser.add_argument('--iterative_on',choices=["section","step","both"])
parser.add_argument('--c', default=100)
args = parser.parse_args()

pe_folder = config.get("SETTINGS", "pe_folder")
counterfactual_path = config.get("SETTINGS", "counterfactual_path")

conf = args.mode
model = build_dnn()
model.load_weights("models/DNN_w.h5")

if conf != "load_counterfactual":
    # CREATE DATASET FROM PE
    if conf == "generate_dataset":
        ds = create_dataset(pe_folder)
        x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
        del ds
    elif conf == "load_dataset":
        # LOAD EXTRACTED LIEF DATASET
        with open(config.get("SETTINGS", "lief_dataset"), 'rb') as handle:
            ds = pickle.load(handle)
        x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
        del ds
    # GENERATE COUNTERFACTUALS

    data = generate_counterfactual(x_train, x_val, x_test, y_train, y_val, y_test, model,
                                   offset_min=int(args.offsetmin), offset_max=int(args.offsetmax))
else:
    # LOAD ALREADY GENERATED CONTERFACTUALS
    with open(counterfactual_path, "rb") as h:
        data = pickle.load(h)

    with open(config.get("SETTINGS", "lief_dataset"), 'rb') as handle:
        ds = pickle.load(handle)
    x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
    del ds

mode = "OneShot"

STEPS = int(args.eta)
SECTIONS = int(args.section)
if args.iterative == "1":
    if args.iterative_on == "section":
        mode = "IterOnSection"
    elif args.iterative_on == "step":
        mode = "IterOnStep"
    else:
        mode = "IterAll"

# FIND ALREADY GENERATED ADVERSARIAL EXAMPLES TO NOT REPEAT THE COMPUTATION
todo = list(data.keys())[:]
print(f"[+] Total Counterfactuals to Analyze: {len(todo)}")
dir_output = "results/" + str(mode) + "_step" + str(STEPS) + "_sec" + str(SECTIONS) + "c" + str(args.c) + "_"+str(args.injection_type)+"/"
try:
    os.mkdir(dir_output)
except:
    pass
# ONESHOT IS THE DEFAULT, OTHER CONFIGURATIONS ALLOW TO USE AN LOGICAL "OR" FOR STEPS AND SECTION PARAMETERS
if mode == "OneShot":
    for d in os.listdir(dir_output):
        if ".json" in d:
            try:
                to_delete = int(d.split("-")[0])
                index = int(np.where(np.array(todo) == to_delete)[0][0])
                del todo[index]
            except:
                pass
threads = []
time_start = datetime.datetime.now()
while True:
    if len(threads) < num_thread:
        if len(todo) == 0:
            print("[+] DONE")
            break
        else:
            if len(todo) > 0:
                mi_n = todo[0]
                del todo[0]
                print("[+] STARTING\t" + str(mi_n))
                # LOADS INDEXES CONTAINING DIFFERENCES BETWEEN ORIGINAL AND GENERATED CONTERFACTUAL,
                # THE CONTERFACTUAL EXAMPLE, THE ORIGINAL LIEF EXAMPLE AND THE TIME SPENT ON GENERATION
                found, not_found, differences, differences_index, target, test, times = data[mi_n]
                # FIND THE FILENAME OF THE ORIGINAL PE FILE
                index_file1 = find_id(test[0][0], x_test_meta)
                if index_file1 is -1:
                    index_file1 = find_id(test[0][0], x_train_meta)
                if index_file1 is -1:
                    index_file1 = find_id(test[0][0], x_val_meta)
                if ("temp-" + str(mi_n) + "-adv.exe") in dir_output:
                    resume = True
                else:
                    resume = False

                p1 = threading.Thread(target=adv_step_mode, args=(mode,
                                                                  differences_index, target, index_file1, model, mi_n,
                                                                  STEPS, dir_output, SECTIONS,
                                                                  pe_folder, False, OPTIMIZED, int(args.c),args.injection_type))

                threads.append(p1)
                p1.start()
            else:
                pass
    threads = [t for t in threads if t.is_alive()]
    # time.sleep(5)

for t in threads:
    t.join()

time_end = datetime.datetime.now()
print((time_end - time_start).seconds)
f = open(dir_output + "time.txt", "w")
f.write(str((time_end - time_start).seconds))
f.close()
print("[+] ENDED")
