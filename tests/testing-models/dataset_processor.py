from utils import activate_filter, get_data_and_labels_at

from skfp.datasets.moleculenet import load_ogb_splits


class DatasetProcessor:
    def __init__(self, dataset_name, data, labels):
        self.dataset_name = dataset_name
        self.data = data
        self.labels = labels
        self.train_idx, self.valid_idx, self.test_idx = load_ogb_splits(dataset_name)
        self.X_train, self.y_train = get_data_and_labels_at(
            data, labels, self.train_idx
        )
        self.X_valid, self.y_valid = get_data_and_labels_at(
            data, labels, self.valid_idx
        )
        self.X_test, self.y_test = get_data_and_labels_at(data, labels, self.test_idx)

    def get_filtered_data(self, filter_fn=None):
        if filter_fn:
            if not isinstance(filter_fn, list):
                filter_fn = [filter_fn]

            X_train, y_train = self.X_train, self.y_train
            X_valid, y_valid = self.X_valid, self.y_valid
            X_test, y_test = self.X_test, self.y_test

            for fn in filter_fn:
                X_train, y_train, X_valid, y_valid, X_test, y_test = activate_filter(
                    fn, X_train, y_train, X_valid, y_valid, X_test, y_test
                )

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        return (
            self.X_train,
            self.y_train,
            self.X_valid,
            self.y_valid,
            self.X_test,
            self.y_test,
        )
