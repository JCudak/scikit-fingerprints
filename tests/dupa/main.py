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
from skfp.fingerprints import (
    AtomPairFingerprint,
    AutocorrFingerprint,
    AvalonFingerprint,
    E3FPFingerprint,
    ECFPFingerprint,
    ElectroShapeFingerprint,
    ERGFingerprint,
    EStateFingerprint,
    FunctionalGroupsFingerprint,
    GETAWAYFingerprint,
    GhoseCrippenFingerprint,
    KlekotaRothFingerprint,
    LaggnerFingerprint,
    LayeredFingerprint,
    LingoFingerprint,
    MACCSFingerprint,
    MAPFingerprint,
    MHFPFingerprint,
    MordredFingerprint,
    MORSEFingerprint,
    MQNsFingerprint,
    PatternFingerprint,
    PharmacophoreFingerprint,
    PhysiochemicalPropertiesFingerprint,
    PubChemFingerprint,
    RDFFingerprint,
    RDKit2DDescriptorsFingerprint,
    RDKitFingerprint,
    SECFPFingerprint,
    TopologicalTorsionFingerprint,
    USRFingerprint,
    USRCATFingerprint,
    VSAFingerprint,
    WHIMFingerprint
)
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


def smiles_to_fingerprint(smiles, fingerprint_class):
    fingerprint = fingerprint_class()

    return fingerprint.transform(smiles)


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
    "No Filter": None,
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

fingerprint_classes = {
    "AtomPairFingerprint": AtomPairFingerprint,
    "AutocorrFingerprint": AutocorrFingerprint,
    "AvalonFingerprint": AvalonFingerprint,
    # "E3FPFingerprint": E3FPFingerprint, OUT
    "ECFPFingerprint": ECFPFingerprint,
    # "ElectroShapeFingerprint": ElectroShapeFingerprint, OUT
    "ERGFingerprint": ERGFingerprint,
    "EStateFingerprint": EStateFingerprint,
    "FunctionalGroupsFingerprint": FunctionalGroupsFingerprint,
    # "GETAWAYFingerprint": GETAWAYFingerprint, OUT
    "GhoseCrippenFingerprint": GhoseCrippenFingerprint,
    "KlekotaRothFingerprint": KlekotaRothFingerprint,
    "LaggnerFingerprint": LaggnerFingerprint,
    "LayeredFingerprint": LayeredFingerprint,
    "LingoFingerprint": LingoFingerprint,
    "MACCSFingerprint": MACCSFingerprint,
    "MAPFingerprint": MAPFingerprint,
    "MHFPFingerprint": MHFPFingerprint,
    "MordredFingerprint": MordredFingerprint,
    # "MORSEFingerprint": MORSEFingerprint, OUT
    "MQNsFingerprint": MQNsFingerprint,
    "PatternFingerprint": PatternFingerprint,
    "PharmacophoreFingerprint": PharmacophoreFingerprint,
    "PhysiochemicalPropertiesFingerprint": PhysiochemicalPropertiesFingerprint,
    "PubChemFingerprint": PubChemFingerprint,
    # "RDFFingerprint": RDFFingerprint, OUT
    "RDKit2DDescriptorsFingerprint": RDKit2DDescriptorsFingerprint,
    "RDKitFingerprint": RDKitFingerprint,
    "SECFPFingerprint": SECFPFingerprint,
    "TopologicalTorsionFingerprint": TopologicalTorsionFingerprint,
    # "USRFingerprint": USRFingerprint, OUT
    # "USRCATFingerprint": USRCATFingerprint, OUT
    "VSAFingerprint": VSAFingerprint,
    # "WHIMFingerprint": WHIMFingerprint OUT
}

forbidden_datasets = ["MUV", "Tox21", "ToxCast", "PCBA"]

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

    def process(self, train_X, train_y, test_X, test_y, fingerprint_class):
        fingerprints_train = smiles_to_fingerprint(train_X, fingerprint_class)
        fingerprints_test = smiles_to_fingerprint(test_X, fingerprint_class)

        model = self.model_factory()
        model.fit(fingerprints_train, train_y)

        y_pred = model.predict(fingerprints_test)
        return calculate_accuracy(y_pred, test_y)




def check_combinations(processor, model_pipeline, dataset_name, results):
    accuracy_difference = 0
    train_X, train_y, valid_X, valid_y, test_X, test_y = (
        processor.get_filtered_data(None)
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

        for filter_names in combinations(["rule_of_3", "faf4_druglike", "faf4_leadlike", "tice_insecticides", "hao",
                                          "tice_herebicides", "lint"], 2):
            filter_fns = [filter_dict[filter_name] for filter_name in filter_names]

            train_X, train_y, valid_X, valid_y, test_X, test_y = (
                processor.get_filtered_data(filter_fns)
            )

            accuracy = model_pipeline.process(
                train_X, train_y, test_X, test_y, fingerprint_class
            )
            accuracy_difference = calculate_accuracy_difference(
                accuracy, base_accuracy
            )

            results.append(
                {
                    "dataset_name": dataset_name,
                    "fingerprint_name": fingerprint_name,
                    "filter_name": filter_names,
                    "filter_accuracy": accuracy,
                    "filter_difference": accuracy_difference,
                }
            )
            print_results(fingerprint_name, filter_names, base_accuracy, accuracy_difference)


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

            accuracy_difference = calculate_accuracy_difference(
                accuracy, base_accuracy
            )

            results.append(
                {
                    "dataset_name": dataset_name,
                    "fingerprint_name": fingerprint_name,
                    "filter_name": filter_name,
                    "filter_accuracy": accuracy,
                    "filter_difference": accuracy_difference,
                }
            )
            print_results(fingerprint_name, filter_name, base_accuracy, accuracy_difference)


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


def print_results(fingerprint_name, filter_name, accuracy, accuracy_difference):
    """Displays info"""
    print(f"\nFingerprint '{fingerprint_name}'")
    print(f"Filter '{filter_name}': Accuracy = {accuracy:.4f}")
    if filter_name != "None":
        print(f"The difference for {filter_name} is: {accuracy_difference:.3f}")



def main():
    combination_results = []
    filter_results = []
    for dataset in datasets:
        dataset_name, data, labels = dataset
        if dataset_name in forbidden_datasets:
            continue
        print(f"Processing dataset: {dataset_name}")

        processor = DatasetProcessor(dataset_name, data, labels)
        model_pipeline = ModelPipeline(create_model)

        check_combinations(processor, model_pipeline, dataset_name, combination_results)

        save_results_to_json(combination_results, filename="combination_results.json")

    for dataset in datasets:
        dataset_name, data, labels = dataset
        if dataset_name in forbidden_datasets:
            continue
        print(f"Processing dataset: {dataset_name}")

        processor = DatasetProcessor(dataset_name, data, labels)
        model_pipeline = ModelPipeline(create_model)

        check_filters(processor, model_pipeline, dataset_name, filter_results)

        save_results_to_json(filter_results, filename="new_filter_results.json")


if __name__ == "__main__":
    main()
