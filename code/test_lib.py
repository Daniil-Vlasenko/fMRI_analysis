import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def accuracy_weight(data_per_tr_file, data_im_tr_file, data_per_test_file, data_im_test_file):
    data_per_tr = pd.read_csv(data_per_tr_file)
    data_im_tr = pd.read_csv(data_im_tr_file)
    data_per_test = pd.read_csv(data_per_test_file)
    data_im_test = pd.read_csv(data_im_test_file)

    data_per_tr = data_per_tr[["sum", "mean", "quantile_1lt", "quantile_2gt", "std"]]
    data_im_tr = data_im_tr[["sum", "mean", "quantile_1lt", "quantile_2gt", "std"]]
    data_per_test = data_per_test[["sum", "mean", "quantile_1lt", "quantile_2gt", "std"]]
    data_im_test = data_im_test[["sum", "mean", "quantile_1lt", "quantile_2gt", "std"]]

    data_tr = pd.concat([data_per_tr, data_im_tr])
    data_test = pd.concat([data_per_test, data_im_test])

    Y_tr = [1 for i in range(len(data_per_tr))] + [2 for i in range(len(data_im_tr))]
    Y_test = [1 for i in range(len(data_per_test))] + [2 for i in range(len(data_im_test))]

    svc = SVC(kernel="rbf", C=10, probability=True)
    svc.fit(data_tr, Y_tr)
    Y_pred = svc.predict(data_test)

    return accuracy_score(Y_pred, Y_test)
