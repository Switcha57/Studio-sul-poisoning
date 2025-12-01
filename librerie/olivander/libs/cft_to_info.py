import json
import numpy as np


def cft_to_info(e1, range=256):
    cft_meta = json.loads(e1.to_json())
    test = cft_meta["test_data"]
    cft = cft_meta["cfs_list"]
    output = []
    original=[]
    found = 0
    not_found = 0
    differences = []
    differences_index = []
    for i, x in enumerate(test):
        if cft[i] is None:
            not_found = not_found + 1
            output.append([0])
            original.append([0])
            differences.append([0])
            differences_index.append([0])
        else:
            found = found + 1
            output.append([])
            differences.append([])
            differences_index.append([])
            original.append([])

            for j, y in enumerate(cft[i]):
                output[i].append(y)
                original[i].append(test[i][0])
                d = np.where(np.array(test[i][0]) != np.array(y))
                differences_index[i] = d
                differences[i].append([np.array(x[0])[d], np.array(y)[d]])
    return found,not_found,differences,differences_index,output,original
#  differences=[test]
