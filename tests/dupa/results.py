import json

def save_results_to_json(all_results, filename):
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"All results saved to '{filename}'.")


def calculate_accuracy_difference(accuracy_with_filter, accuracy_without_filter):
    return accuracy_with_filter - accuracy_without_filter


def print_results(fingerprint_name, filter_name, accuracy, accuracy_difference):
    print(f"\nFingerprint '{fingerprint_name}'")
    print(f"Filter '{filter_name}': Accuracy = {accuracy:.4f}")
    if filter_name != "None":
        print(f"The difference for {filter_name} is: {accuracy_difference:.3f}")
