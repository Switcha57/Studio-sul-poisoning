import os

import numpy as np
import pandas as pd
from sklearn import metrics
from art.attacks.evasion import FastGradientMethod
from sklearn.model_selection import train_test_split
from art.estimators.classification import KerasClassifier

from libs.build_dnn import build_dnn

th = [["conf", "ds", "tn", "fp", "fn", "tp"]]


def cm_to_excel(conf, ds, cm):
    if len(cm) == 2:
        t = [conf, ds, cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
    elif cm == [[100]]:
        t = [conf, ds, 0, 0, 0, 100]
    else:
        t=[]
        print("error")
    return t


def adv_scores(model, x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta,
               eps, my_100, y_100):
    print("[+] adv_train..")
    output_dir = "results/adv_train/"
    try:
        os.mkdir(output_dir)
    except:
        pass
    output_dir = "results/adv_train/" + str(eps) + "/"
    try:
        os.mkdir(output_dir)
    except:
        print("folder already created")

    ypred = model.predict(x_train)
    ypred = np.argmax(ypred, axis=-1)

    cm_original = metrics.confusion_matrix(y_train, ypred)
    print_original = metrics.classification_report(y_train, ypred)

    ypred = model.predict(x_test)
    ypred = np.argmax(ypred, axis=-1)
    cm_original_test = metrics.confusion_matrix(y_test, ypred)
    print_original_test = metrics.classification_report(y_test, ypred)

    m = KerasClassifier(model=model)
    attack = FastGradientMethod(estimator=m, eps=eps, num_random_init=0)
    x_train_adv = attack.generate(x=x_train)

    ypred = model.predict(x_train_adv)
    ypred = np.argmax(ypred, axis=-1)

    cm_adv = metrics.confusion_matrix(y_train, ypred)
    print_adv = metrics.classification_report(y_train, ypred)

    from keras.callbacks import EarlyStopping
    e = EarlyStopping(monitor="val_loss", patience=10)

    new_xtrain = np.concatenate((x_train, x_train_adv), axis=0)
    new_ytrain = np.concatenate((y_train, y_train), axis=0)

    new_xtrain, new_xval, new_ytrain, new_yval = train_test_split(new_xtrain, new_ytrain, test_size=0.1,
                                                                  random_state=0, stratify=new_ytrain)
    hist = model.fit(new_xtrain, new_ytrain, validation_data=(new_xval, new_yval), epochs=2000, batch_size=256,
                     callbacks=[e])
    model.save_weights(output_dir + "model_adv_" + str(eps) + ".tf")

    ypred = model.predict(new_xtrain)
    ypred = np.argmax(ypred, axis=-1)
    new_model_ontrain = metrics.confusion_matrix(new_ytrain, ypred)
    print_new_model_ontrain = metrics.classification_report(new_ytrain, ypred)

    ypred = model.predict(x_train_adv)
    ypred = np.argmax(ypred, axis=-1)

    new_model_adv = metrics.confusion_matrix(y_train, ypred)
    print_new_model_adv = metrics.classification_report(y_train, ypred)

    ypred = model.predict(my_100)
    ypred = np.argmax(ypred, axis=-1)

    new_model_res_100 = metrics.confusion_matrix(y_100, ypred)
    print_new_model_res_100 = metrics.classification_report(y_100, ypred)

    ypred = model.predict(x_test)
    ypred = np.argmax(ypred, axis=-1)

    new_model_test = metrics.confusion_matrix(y_test, ypred)
    print_new_model_test = metrics.classification_report(y_test, ypred)

    print()
    res = [model, [cm_original, print_original, cm_original_test, print_original_test, cm_adv
        , print_adv, new_model_ontrain, print_new_model_ontrain, new_model_adv, print_new_model_adv,
                   new_model_res_100, print_new_model_res_100, new_model_test, print_new_model_test]]

    res_name = ["original_model_train", "original_model_train_report", "original_model_test",
                "original_model_test_report", "original_model_fgsm(train)"
        , "original_model_fgsm(train)_report", "adv_training_model_train", "adv_training_model_train_report",
                "adv_training_model_fgsm(train)", "adv_training_model_fgsm(train)_report",
                "adv_training_model_100_examples", "adv_training_model_100_examples_report", "adv_training_model_test",
                "adv_training_model_test_report"]

    for i, e in enumerate(res[1]):
        with open(output_dir + 'scores_' + str(eps) + '.txt', 'a') as f:
            print(res_name[i], file=f)
            print(e, file=f)
            print("\n", file=f)

    new_excel = [["conf", "ds", "tn", "fp", "fn", "tp"]]
    new_excel.append(cm_to_excel("original_model", "train", cm_original))
    new_excel.append(cm_to_excel("original_model", "test", cm_original_test))
    new_excel.append(cm_to_excel("original_model", "fgsm(train)", cm_adv))
    new_excel.append(cm_to_excel("adv_training_model", "train", new_model_ontrain))
    new_excel.append(cm_to_excel("adv_training_model", "fgsm(train)", new_model_adv))
    new_excel.append(cm_to_excel("adv_training_model", "100 examples", new_model_res_100))
    new_excel.append(cm_to_excel("adv_training_model", "test", new_model_test))
    pd.DataFrame(new_excel).to_excel(output_dir + str(eps) + ".xlsx")
    return [model, [cm_original, print_original, cm_original_test, print_original_test, cm_adv
        , print_adv, new_model_ontrain, print_new_model_ontrain, new_model_adv, print_new_model_adv,
                    new_model_res_100, print_new_model_res_100, new_model_test, print_new_model_test]]
