from sklearn.ensemble import RandomForestClassifier
from utils import smiles_to_fingerprint
import numpy as np

class ModelPipeline:
    def __init__(self, model_factory):
        self.model_factory = model_factory

    def process(self, train_X, train_y, test_X, test_y, fingerprint_class):
        fingerprints_train = smiles_to_fingerprint(train_X, fingerprint_class)
        fingerprints_test = smiles_to_fingerprint(test_X, fingerprint_class)

        model = self.model_factory()
        model.fit(fingerprints_train, train_y)

        y_pred = model.predict(fingerprints_test)
        return np.mean(y_pred == test_y)


def create_model():
    hyperparams = {
        "n_estimators": 1000,
        "criterion": "entropy",
        "min_samples_split": 10,
    }
    return RandomForestClassifier(**hyperparams, n_jobs=-1, random_state=0, verbose=False)
