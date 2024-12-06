def get_data_and_labels_at(data, labels, indexes):
    split_data = [data[i] for i in indexes]
    split_labels = [labels[i] for i in indexes]
    return split_data, split_labels


def smiles_to_fingerprint(smiles, fingerprint_class):
    fingerprint = fingerprint_class()
    return fingerprint.transform(smiles)


def activate_filter(filter, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train_filtered, y_train_filtered = filter_x_and_y(X_train, y_train, filter)
    X_test_filtered, y_test_filtered = filter_x_and_y(X_test, y_test, filter)
    X_valid_filtered, y_valid_filtered = filter_x_and_y(X_valid, y_valid, filter)
    return X_train_filtered, y_train_filtered, X_valid_filtered, y_valid_filtered, X_test_filtered, y_test_filtered


def filter_x_and_y(data, labels, filter):
    filtered = [(x, labels[i]) for i, x in enumerate(data) if filter.transform([x])]
    if filtered:
        filtered_data, filtered_labels = zip(*filtered)
        return list(filtered_data), list(filtered_labels)
    else:
        return [], []
