import configparser
import datetime as datetime
import os

from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi
from secml_malware.models import CClassifierEmber
from libs.Gamma_wrapper.custom_c_classifier_ember import CustomClassifierEmber
import pickle
from libs.Gamma_wrapper.CustomClassifier import CustomClassifier
import numpy as np
from secml.array import CArray
from libs.build_dnn import build_dnn
from libs.dataset_to_features import dataset_to_features
from libs.find_id import find_id

np.int = np.int64
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm

from secml_malware.attack.blackbox.c_gamma_sections_evasion import CGammaSectionsEvasionProblem
from secml_malware.attack.blackbox.c_gamma_evasion import CGammaEvasionProblem
from secml.ml.classifiers import CClassifier


config = configparser.ConfigParser()
config.read('config.ini')
CClassifierEmber.__init__=CustomClassifierEmber.__init__
CClassifierEmber.extract_features=CustomClassifierEmber.extract_features
CClassifierEmber._forward=CustomClassifierEmber._forward

CClassifier._check_is_fitted=CustomClassifier._check_is_fitted

model = build_dnn()
model.load_weights("models/DNN_w.h5")
net = CClassifierEmber(model)
net = CEmberWrapperPhi(net)

results = []

X = []
X_byte = []
y = []
file_names = []
dir = config.get("SETTINGS", "pe_folder")

# dataset_file_meta = 'pickle/dataset_features_meta.pickle'
counterfactual_path = config.get("SETTINGS", "counterfactual_path")
with open(counterfactual_path, "rb") as h:
    data = pickle.load(h)
with open(config.get("SETTINGS", "lief_dataset"), 'rb') as handle:
    ds = pickle.load(handle)
x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
del ds

goodware_folder = config.get("GAMMA", "goodware_folder")

starting = datetime.datetime.now()
res = {}

section_population, what_from_who = CGammaEvasionProblem.create_section_population_from_folder(goodware_folder,
                                                                                               how_many=10,
                                                                                               sections_to_extract=[
                                                                                                   '.rdata'])
if config.get("GAMMA", "gamma_manipulation_type") == "padding":
    attack = CGammaEvasionProblem(section_population, net, population_size=50, penalty_regularizer=1e-6,
                                  iterations=5000, threshold=0, )
else:
    attack = CGammaSectionsEvasionProblem(section_population, net, population_size=10, penalty_regularizer=1e-6,
                                          iterations=1000, threshold=0, seed=0)
engine = CGeneticAlgorithm(attack)

dir_output = "results/gamma_" + str(config.get("GAMMA", "gamma_manipulation_type")) + "/"
try:
    os.mkdir(dir_output)
except:
    print("folder already created")
    pass

for mi in data.keys():
    found, not_found, differences, differences_index, target, test, times = data[mi]
    index_file1 = find_id(test[0][0], x_test_meta)
    with open(dir + "samples/" + str(index_file1), "rb") as file_handle:
        code = file_handle.read()
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, confidence = net.predict(CArray(x), True)
    results.append({"name": index_file1, "id": mi, "original": confidence[0, 1].item()})
    X.append(x)
    X_byte.append(code)
    conf = confidence[1][0].item()
    y.append([1 - conf, conf])

    res[results[-1]["id"]] = {}
    res[results[-1]["id"]]["partial"] = False
    res[results[-1]["id"]]["final"] = False
    label = y[-1]
    print("Starting example id:" + str(results[-1]["id"]))
    start = datetime.datetime.now()
    y_pred, adv_score, adv_ds, f_obj = engine.run(x, CArray(label[1]))
    stop = datetime.datetime.now()
    res[results[-1]["id"]]["time"] = (stop - start).seconds
    if y_pred[0] == 0:
        res[results[-1]["id"]]["partial"] = True
        adv_x = adv_ds.X[0, :]
        engine.write_adv_to_file(adv_x, dir_output + str(results[-1]["id"]) + 'adv_exe')
        with open(dir_output + str(results[-1]["id"]) + 'adv_exe', 'rb') as h:
            code = h.read()
        real_adv_x = CArray(np.frombuffer(code, dtype=np.uint8))
        _, confidence = net.predict(CArray(real_adv_x), True)
        if confidence[0][0] < confidence[1][0]:
            print("error")
        else:
            res[results[-1]["id"]]["final"] = confidence[0][0]
            engine.write_adv_to_file(adv_x, dir_output + "final-" + str(results[-1]["id"]) + 'adv_exe')
            print("Evaded id:" + str(results[-1]["id"]))

        print(confidence[0, 1].item())
ending = datetime.datetime.now()
print("time:" + str((ending - starting).seconds))
with open(dir_output + "res.pickle", 'wb') as h:
    pickle.dump(res, h)
print("--")
