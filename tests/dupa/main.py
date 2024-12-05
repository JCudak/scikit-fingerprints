import json
from itertools import combinations

import numpy as np
from ogb.graphproppred import Evaluator
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

from skfp.datasets.moleculenet import load_moleculenet_benchmark, load_ogb_splits
from skfp.filters import (
    BeyondRo5Filter,
    BMSFilter,
    BrenkFilter,
    FAF4DruglikeFilter,
    FAF4LeadlikeFilter,
    GhoseFilter,
    HaoFilter,
    InpharmaticaFilter,
    LINTFilter,
    LipinskiFilter,
    MLSMRFilter,
    MolecularWeightFilter,
    NIBRFilter,
    NIHFilter,
    OpreaFilter,
    PAINSFilter,
    PfizerFilter,
    REOSFilter,
    RuleOfFourFilter,
    RuleOfThreeFilter,
    RuleOfTwoFilter,
    RuleOfXuFilter,
    SureChEMBLFilter,
    TiceHerbicidesFilter,
    TiceInsecticidesFilter,
    ValenceDiscoveryFilter,
    ZINCBasicFilter,
    ZINCDruglikeFilter,
)
from skfp.fingerprints import AtomPairFingerprint
from skfp.preprocessing import MolFromSmilesTransformer

"""
TO NA DOLE
TO POWINNO WYLACZAC WARNINGI ALE NWM 

"""
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

datasets = load_moleculenet_benchmark(subset="classification")


def get_data_and_labels_at(data, labels, indexes):
    split_data = [data[i] for i in indexes]
    split_labels = [labels[i] for i in indexes]

    return split_data, split_labels


def filter_x_and_y(data, labels, filter):
    filtered = [
        (x, labels[i]) for i, x in enumerate(data) if filter.transform([x])
    ]

    if filtered:
        filtered_data, filtered_labels = zip(*filtered)
        return list(filtered_data), list(filtered_labels)
    else:
        return [], []


def smiles_to_fingerprint(smiles):
    atom_pair_fingerprint = AtomPairFingerprint()

    X = atom_pair_fingerprint.transform(smiles)
    return X


def get_model(
        random_state: int,
        hyperparams: dict,
        verbose: bool,
):
    n_jobs = -1

    model = RandomForestClassifier(
        **hyperparams,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    return model


def evaluate_model(
        dataset_name: str,
        task_type: str,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> float:
    # use OGB evaluation for MoleculeNet
    if task_type == "classification":
        y_pred = model.predict_proba(X_test)[:, 1]
        y_test = y_test.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    elif task_type == "multioutput_classification":
        # extract positive class probability for each task
        y_pred = model.predict_proba(X_test)
        y_pred = [y_pred_i[:, 1] for y_pred_i in y_pred]
        y_pred = np.column_stack(y_pred)
    else:
        raise ValueError(f"Task type '{task_type}' not recognized")

    evaluator = Evaluator(dataset_name)
    metrics = evaluator.eval(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    # extract the AUROC
    metric = next(iter(metrics.values()))
    return metric


def activate_filter(filter, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train_filtered, y_train_filtered = filter_x_and_y(X_train, y_train, filter)
    X_test_filtered, y_test_filtered = filter_x_and_y(X_test, y_test, filter)
    X_valid_filtered, y_valid_filtered = filter_x_and_y(X_valid, y_valid, filter)

    return (
        X_train_filtered,
        y_train_filtered,
        X_valid_filtered,
        y_valid_filtered,
        X_test_filtered,
        y_test_filtered,
    )


filter_dict = {
    "None": None,
    "Lipinski": LipinskiFilter(),
    "BeyondRo5": BeyondRo5Filter(),
    "BMS": BMSFilter(),
    "Brenk": BrenkFilter(),
    "faf4_druglike": FAF4DruglikeFilter(),
    "faf4_leadlike": FAF4LeadlikeFilter(),
    "ghose": GhoseFilter(),
    "hao": HaoFilter(),
    "inpharmatica": InpharmaticaFilter(),
    "lint": LINTFilter(),
    "mlsmr": MLSMRFilter(),
    "mol_weight": MolecularWeightFilter(),
    "nibr": NIBRFilter(),
    "nih": NIHFilter(),
    "oprea": OpreaFilter(),
    "pains": PAINSFilter(),
    "pfizer": PfizerFilter(),
    # "reos": REOSFilter(),
    # "rule_of_2" : RuleOfTwoFilter(),
    "rule_of_3": RuleOfThreeFilter(),
    "rule_of_4": RuleOfFourFilter(),
    "rule_of_xu": RuleOfXuFilter(),
    "surechembl": SureChEMBLFilter(),
    "tice_herebicides": TiceHerbicidesFilter(),
    "tice_insecticides": TiceInsecticidesFilter(),
    "valence_discovery": ValenceDiscoveryFilter(),
    "zinc_basic": ZINCBasicFilter(),
    "zinc_druglike": ZINCDruglikeFilter(),
}


class DatasetProcessor:
    """
    This class is respobsible for dividing the dataset into train, test and validation sets.


    """

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


class ModelPipeline:
    def __init__(self, model_factory):
        self.model_factory = model_factory

    def process(self, train_X, train_y, test_X, test_y):
        fingerprints_train = smiles_to_fingerprint(train_X)
        fingerprints_test = smiles_to_fingerprint(test_X)

        model = self.model_factory()
        model.fit(fingerprints_train, train_y)

        y_pred = model.predict(fingerprints_test)
        return calculate_accuracy(y_pred, test_y)

def check_combinations():
    all_results = []
    for dataset in datasets:
        dataset_name, data, labels = dataset
        if dataset_name in ["MUV", "Tox21", "ToxCast", "PCBA"]:
            continue
        print(f"Processing dataset: {dataset_name}")

        processor = DatasetProcessor(dataset_name, data, labels)
        model_pipeline = ModelPipeline(create_model)
        accuracy_difference = 0
        train_X, train_y, valid_X, valid_y, test_X, test_y = (
            processor.get_filtered_data(None)
        )
        accuracy_without_filter = model_pipeline.process(
            train_X, train_y, test_X, test_y
        )
        all_results.append(
            {
                "dataset_name": dataset_name,
                "filter_name": "No Filter",
                "filter_accuracy": accuracy_without_filter,
                "filter_difference": accuracy_difference,
            }
        )

        print_results("None", accuracy_without_filter, accuracy_difference)

        for filter_names in combinations(["rule_of_3", "faf4_druglike", "faf4_leadlike", "tice_insecticides", "hao",
                                          "tice_herebicides", "lint"], 2):
            filter_fns = [filter_dict[filter_name] for filter_name in filter_names]

            train_X, train_y, valid_X, valid_y, test_X, test_y = (
                processor.get_filtered_data(filter_fns)
            )

            accuracy_with_filter = model_pipeline.process(
                train_X, train_y, test_X, test_y
            )
            accuracy_difference = calculate_accuracy_difference(
                accuracy_with_filter, accuracy_without_filter
            )

            all_results.append(
                {
                    "dataset_name": dataset_name,
                    "filter_name": filter_names,
                    "filter_accuracy": accuracy_with_filter,
                    "filter_difference": accuracy_difference,
                }
            )
            print_results(filter_names, accuracy_without_filter, accuracy_difference)
        save_results_to_json(all_results, filename="filter_results2.json")

def main():
    all_results = []  # this list collects all results for every dataset

    for dataset in datasets:
        dataset_name, data, labels = dataset
        if dataset_name in ["MUV", "Tox21", "ToxCast", "PCBA"]:
            continue
        print(f"Processing dataset: {dataset_name}")

        processor = DatasetProcessor(dataset_name, data, labels)
        model_pipeline = ModelPipeline(create_model)

        for filter_name, filter_fn in filter_dict.items():
            train_X, train_y, valid_X, valid_y, test_X, test_y = (
                processor.get_filtered_data(filter_fn)
            )

            if filter_fn is None:
                accuracy_difference = 0
                accuracy_without_filter = model_pipeline.process(
                    train_X, train_y, test_X, test_y
                )
                all_results.append(
                    {
                        "dataset_name": dataset_name,
                        "filter_name": "No Filter",
                        "filter_accuracy": accuracy_without_filter,
                        "filter_difference": accuracy_difference,
                    }
                )
                print_results(filter_name, accuracy_without_filter, accuracy_difference)
                continue

            accuracy_with_filter = model_pipeline.process(
                train_X, train_y, test_X, test_y
            )
            accuracy_difference = calculate_accuracy_difference(
                accuracy_with_filter, accuracy_without_filter
            )

            all_results.append(
                {
                    "dataset_name": dataset_name,
                    "filter_name": filter_name,
                    "filter_accuracy": accuracy_with_filter,
                    "filter_difference": accuracy_difference,
                }
            )
            print_results(filter_name, accuracy_without_filter, accuracy_difference)
        # Save all results to a single JSON file
        save_results_to_json(all_results, filename="filter_results2.json")


def save_results_to_json(all_results, filename):
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"All results saved to '{filename}'.")


def calculate_accuracy_difference(accuracy_with_filter, accuracy_without_filter):
    return accuracy_with_filter - accuracy_without_filter


def create_model():
    """Creates model with pre-defined hyperparameters."""
    hyperparams = {
        "n_estimators": 1000,
        "criterion": "entropy",
        "min_samples_split": 10,
    }
    return get_model(random_state=0, hyperparams=hyperparams, verbose=False)


def calculate_accuracy(y_pred, y_true):
    """Calculate the accuracy"""
    return np.mean(y_pred == y_true)


def print_results(filter_name, accuracy, accuracy_difference):
    """Displays info"""
    if filter_name == "None":
        print(f"Filter '{filter_name}': Accuracy = {accuracy:.4f}")
    else:
        print(f"Filter '{filter_name}': Accuracy = {accuracy:.4f}")
        print(f"The difference for {filter_name} is: {accuracy_difference:.3f}")


### TODO:
# możemy pokazać to jeszcze w df, csv i nwm
# generalnie jeszcze nie wiem dlaczgo te datasety nie działaja


def create_df(json_path):
    import pandas as pd

    df = pd.read_json(json_path)
    return df


if __name__ == "__main__":
    # main()
    check_combinations()
    # print(create_df("filter_results.json"))
