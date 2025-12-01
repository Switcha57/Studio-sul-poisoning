from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def dataset_to_features(ds, save=False, output_file="dataset_features.pickle",output_file_meta="dataset_features_meta.pickle"):
    x_train_meta = []
    y_train = []
    x_test_meta = []
    y_test = []

    for i, pe in enumerate(ds):
        x_train_meta.append(pe)
        y_train.append(0 if pe["list"] == "Whitelist" else 1)
    x_train_meta, x_test_meta, y_train, y_test = train_test_split(x_train_meta, y_train, test_size=0.2,
                                                                  random_state=0, stratify=y_train)

    x_train_meta, x_val_meta, y_train, y_val = train_test_split(x_train_meta, y_train, test_size=0.1,
                                                                random_state=0, stratify=y_train)

    x_train = np.array([x["features"] for x in x_train_meta])
    x_val = np.array([x["features"] for x in x_val_meta])
    x_test = np.array([x["features"] for x in x_test_meta])

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    if save:
        with open(output_file, 'wb') as handle:
            pickle.dump([x_train, x_val, x_test, y_train, y_val, y_test], handle)
        with open(output_file_meta, 'wb') as handle:
            pickle.dump([x_train_meta, x_val_meta, x_test_meta], handle)

    return x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta
