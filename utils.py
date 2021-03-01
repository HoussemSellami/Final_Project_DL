import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(all_val_labels, all_val_preds, all_train_labels, all_train_preds):
    all_val_labels = np.concatenate(all_val_labels).reshape(-1, 1)
    all_val_preds = np.concatenate(all_val_preds).reshape(-1, 1)

    all_train_labels = np.concatenate(all_train_labels).reshape(-1, 1)
    all_train_preds = np.concatenate(all_train_preds).reshape(-1, 1)

    acc_val = accuracy_score(all_val_labels, all_val_preds)
    acc_train = accuracy_score(all_train_labels, all_train_preds)

    return acc_val, acc_train


def conf_matrix_test(all_test_labels, all_test_preds):
    return confusion_matrix(all_test_labels, all_test_preds)