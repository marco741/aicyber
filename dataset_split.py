import h5py
import math
import numpy as np
CHOSEN_CLASS = "n004563"  # class with most samples

N_TEST_PERCENTAGE = 0.15
N_VAL_PERCENTAGE = 0.15
FALSE_TRUE_RATIO = 4.0  # for test and validation


OFFSET = 5
N_IDENTITIES = 500

SOURCE_DATA_PATH = "dataset.h5"
DEST_DATA_PATH = "split_dataset.h5"


def n_samples_from_percentage(data_path, percentage, class_name):
    with h5py.File(data_path) as f:
        n = round(f["data"][class_name].shape[0]*percentage)
        return n


def split_once(f, X, y, class_name, start_indices, true_amount, ft_ratio, n_identities):
    X.extend(f["data"][class_name][start_indices[class_name]:start_indices[class_name]+true_amount])
    start_indices[class_name] += true_amount
    y.extend([1]*true_amount)

    n_false_test = round(true_amount * ft_ratio)
    remaining_identities = n_identities - 1
    for dname, d in f["data"].items():
        n_false_test_current = math.ceil(n_false_test/(remaining_identities))
        if class_name != dname:
            X.extend(d[start_indices[dname]:start_indices[dname]+n_false_test_current])
            start_indices[dname] += n_false_test_current
            y.extend([0]*n_false_test_current)
            n_false_test -= n_false_test_current
            remaining_identities -= 1

    return X, y


def split_dataset(data_path, n_test, n_val, class_name, n_identities, ft_ratio, offset=0):
    X_test, y_test = [], []
    X_val, y_val = [], []
    X_train, y_train = [], []

    start_indices = {}
    chosen_class_amount = 0
    with h5py.File(data_path) as f:
        for dname, d in f["data"].items():
            if dname == class_name:
                chosen_class_amount = d.shape[0]
                start_indices[dname] = 0
            else:
                start_indices[dname] = offset

    with h5py.File(data_path) as f:
        X_test, y_test = split_once(f, X_test, y_test, class_name, start_indices, n_test, ft_ratio, n_identities)
        X_val, y_val = split_once(f, X_val, y_val, class_name, start_indices, n_val, ft_ratio, n_identities)
        X_train, y_train = split_once(f, X_train, y_train, class_name, start_indices, chosen_class_amount - start_indices[class_name], 1.0, n_identities)

    return np.array(X_test), np.array(y_test), np.array(X_val), np.array(y_val), np.array(X_train), np.array(y_train)


def write_dataset(data_path, X, y, label):
    with h5py.File(data_path, 'a') as f:
        f.create_dataset(f"{label}/X", data=X)
        f.create_dataset(f"{label}/y", data=y)


n_test_samples = n_samples_from_percentage(SOURCE_DATA_PATH, N_TEST_PERCENTAGE, CHOSEN_CLASS)
n_val_samples = n_samples_from_percentage(SOURCE_DATA_PATH, N_VAL_PERCENTAGE, CHOSEN_CLASS)
X_test, y_test, X_val, y_val, X_train, y_train = split_dataset(
    SOURCE_DATA_PATH, n_test_samples, n_val_samples, CHOSEN_CLASS, N_IDENTITIES, FALSE_TRUE_RATIO, OFFSET)

write_dataset(DEST_DATA_PATH, X_test, y_test, "test")
write_dataset(DEST_DATA_PATH, X_val, y_val, "val")
write_dataset(DEST_DATA_PATH, X_train, y_train, "train")
