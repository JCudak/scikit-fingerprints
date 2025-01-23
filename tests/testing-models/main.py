from config import datasets, forbidden_datasets
from dataset_processor import DatasetProcessor
from evaluation import check_combinations, check_filters
from model_pipeline import ModelPipeline, create_model
from results import save_results_to_json


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
