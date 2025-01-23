from itertools import combinations

from config import filter_dict, fingerprint_classes
from results import calculate_accuracy_difference, print_results


def check_combinations(processor, model_pipeline, dataset_name, results):
    accuracy_difference = 0
    train_X, train_y, valid_X, valid_y, test_X, test_y = processor.get_filtered_data(
        None
    )

    for fingerprint_name, fingerprint_class in fingerprint_classes.items():
        base_accuracy = model_pipeline.process(
            train_X, train_y, test_X, test_y, fingerprint_class
        )
        results.append(
            {
                "dataset_name": dataset_name,
                "fingerprint_name": fingerprint_name,
                "filter_name": "No Filter",
                "filter_accuracy": base_accuracy,
                "filter_difference": accuracy_difference,
            }
        )
        print_results(fingerprint_name, "No Filter", base_accuracy, accuracy_difference)

        for filter_names in combinations(["rule_of_3", "faf4_druglike", "lint"], 2):
            filter_fns = [filter_dict[filter_name] for filter_name in filter_names]
            train_X, train_y, valid_X, valid_y, test_X, test_y = (
                processor.get_filtered_data(filter_fns)
            )

            accuracy = model_pipeline.process(
                train_X, train_y, test_X, test_y, fingerprint_class
            )
            accuracy_difference = calculate_accuracy_difference(accuracy, base_accuracy)

            results.append(
                {
                    "dataset_name": dataset_name,
                    "fingerprint_name": fingerprint_name,
                    "filter_name": filter_names,
                    "filter_accuracy": accuracy,
                    "filter_difference": accuracy_difference,
                }
            )
            print_results(
                fingerprint_name, filter_names, base_accuracy, accuracy_difference
            )


def check_filters(processor, model_pipeline, dataset_name, results):
    base_accuracy = 0
    for fingerprint_name, fingerprint_class in fingerprint_classes.items():
        for filter_name, filter_fn in filter_dict.items():
            train_X, train_y, valid_X, valid_y, test_X, test_y = (
                processor.get_filtered_data(filter_fn)
            )

            accuracy = model_pipeline.process(
                train_X, train_y, test_X, test_y, fingerprint_class
            )
            if filter_fn is None:
                base_accuracy = accuracy

            accuracy_difference = calculate_accuracy_difference(accuracy, base_accuracy)

            results.append(
                {
                    "dataset_name": dataset_name,
                    "fingerprint_name": fingerprint_name,
                    "filter_name": filter_name,
                    "filter_accuracy": accuracy,
                    "filter_difference": accuracy_difference,
                }
            )
            print_results(
                fingerprint_name, filter_name, base_accuracy, accuracy_difference
            )
