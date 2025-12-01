import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
import dice_ml

from libs.cft_to_info import cft_to_info


def generate_counterfactual(x_train, x_val, x_test, y_train, y_val, y_test,
                            model, conterfactual_number=1, offset_min=100, offset_max=200):
    print("[+] Generating Counterfactuals..")
    index_to_map = [str(n) for n in range(x_train.shape[1])]
    df = pd.DataFrame(x_train, columns=index_to_map)
    df_test = pd.DataFrame(x_test, columns=index_to_map)

    # Get predictions on test set
    ypred_test = model.predict(df_test)
    ypred_test = np.argmax(ypred_test, axis=-1)

    print("[+] Test Set Predictions:")
    print(metrics.classification_report(y_test, ypred_test))
    print(metrics.confusion_matrix(y_test, ypred_test))

    # Get predictions on train set
    ypred_train = model.predict(df)
    ypred_train = np.argmax(ypred_train, axis=-1)

    print("[+] Train Set Predictions:")
    print(metrics.classification_report(y_train, ypred_train))
    print(metrics.confusion_matrix(y_train, ypred_train))

    # Extract True Negatives from test set
    negative_test = np.where(ypred_test == 0)[0]
    x_test_true_negative = []
    for p in negative_test:
        if y_test[p] == 0:
            x_test_true_negative.append(x_test[p])

    # Extract True Negatives from train set
    negative_train = np.where(ypred_train == 0)[0]
    x_train_true_negative = []
    for p in negative_train:
        if y_train[p] == 0:
            x_train_true_negative.append(x_train[p])

    # Combine True Negatives from both sets
    x_all_true_negative = np.concatenate([
        np.array(x_train_true_negative),
        np.array(x_test_true_negative)
    ])
    
    df_all_true_negative = pd.DataFrame(
        x_all_true_negative, columns=index_to_map)
    
    print(f"[+] Total True Negatives: {len(x_all_true_negative)} (Train: {len(x_train_true_negative)}, Test: {len(x_test_true_negative)})")

    permitted_range = []
    offset_min = 0
    offset_max = len(df_all_true_negative)
    for i in range(offset_min, offset_max):
        row = {}
        for k in df_all_true_negative.columns[0:256]:
            row[k] = [df_all_true_negative.loc[i, k], 1]
        permitted_range.append(row)

    df_with_label = df
    df_with_label["y"] = y_train

    d = dice_ml.Data(dataframe=df_with_label, outcome_name='y', continuous_features=index_to_map)
    m = dice_ml.Model(model=model, backend="TF2")

    exp = dice_ml.Dice(d, m, method="random")

    res = {}
  
    for i in range(offset_min, offset_max):
        res[i] = {}
        for j in range(conterfactual_number):
            try:
                starting = pd.datetime.datetime.now()
                e1 = exp.generate_counterfactuals(df_all_true_negative.iloc[[i]], total_CFs=1,
                                                  desired_class="opposite",
                                                  features_to_vary=index_to_map[0:256], random_seed=1,
                                                  permitted_range=permitted_range[i - offset_min])
                # e1.visualize_as_dataframe(show_only_changes=True)
                res[i] = [e1, (pd.datetime.datetime.now() - starting).seconds]
                break
            except Exception as e:
                res[i] = [None]
                print(e)

    res_parsed = {}
    for i in res.keys():
        if res[i][0] is not None:
            found, not_found, differences, differences_index, output, test = cft_to_info(res[i][0])
            res_parsed[i] = [found, not_found, differences, differences_index, output, test, res[i][1]]
    with open("pickle/counterfactuals.pickle", 'wb') as h:
        pickle.dump(res_parsed, h)

    return res_parsed
