import pickle

import numpy as np


def find_id(test, x_test_meta):
    t = np.array(test)[0:-1]
    attack_index = -1
    for index, i in enumerate(x_test_meta):
        i2 = i["features"]
        if (i2 == t).all():
            #    print(index)
            attack_index = i["id"]
            #  print(i["id"])
            #  print("ok")
            break
    return attack_index
