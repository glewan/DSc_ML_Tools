from abc import ABC
import preprocessing as prep


class Classifier(ABC):

    def __init__(self):
        pass

    def separate_and_prepare_data(self, data, labels, train_index, test_index, num_features):
        trn_x, tst_x = data[data.index.isin(train_index)], data[data.index.isin(test_index)]
        trn_y, tst_y = labels[labels.index.isin(train_index)], labels[labels.index.isin(test_index)]

        # PREPARATION FOR TRAINING PART
        trn_x, tst_x = prep.normalization(trn_x, tst_x)
        trn_x, trn_y = prep.balance_oversampling(trn_x, trn_y)
        trn_x, tst_x = prep.correlation_removal_kcross(trn_x, tst_x)
        trn_x, tst_x = prep.select_features(trn_x, trn_y, tst_x, num_features)

        trn_x_values = trn_x.values
        tst_x_values = tst_x.values
        trn_y_values = trn_y.values
        tst_y_values = tst_y.values

        return trn_x_values, tst_x_values, trn_y_values, tst_y_values
