import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

from skfp.fingerprints import AtomPairFingerprint
from skfp.model_selection import FingerprintEstimatorGridSearch


class MockClassifier(DummyClassifier):
    def __init__(self, param1: int = 0, param2: int = 0):
        super().__init__(strategy="constant", constant=1)
        self.param1 = param1
        self.param2 = param2


def test_fp_estimator_grid_search(smallest_mols_list):
    num_mols = len(smallest_mols_list)
    pos_mols = np.ones(num_mols // 2)
    neg_mols = np.zeros(num_mols - len(pos_mols))
    y = np.concatenate((pos_mols, neg_mols))
    fp = AtomPairFingerprint()
    fp_param_grid = {"max_distance": [2, 3, 4]}
    estimator_cv = GridSearchCV(
        estimator=MockClassifier(),
        param_grid={"param1": [0, 1, 2], "param2": [0, 1, 2]},
        scoring="accuracy",
    )
    fp_cv = FingerprintEstimatorGridSearch(fp, fp_param_grid, estimator_cv)
    fp_cv.fit(smallest_mols_list, y)

    y_pred = fp_cv.predict(smallest_mols_list)
    y_pred_proba = fp_cv.predict_proba(smallest_mols_list)
    assert len(y_pred) == len(smallest_mols_list)
    assert len(y_pred) == len(y_pred_proba)

    X = fp_cv.transform(smallest_mols_list)
    assert X.shape == (len(smallest_mols_list), fp.n_features_out)

    assert len(fp_cv.cv_results_) == 3
    assert isinstance(fp_cv.best_fp_, AtomPairFingerprint)
    assert isinstance(fp_cv.best_fp_params_, dict)
    assert np.isclose(fp_cv.best_score_, len(pos_mols) / num_mols)

    assert isinstance(fp_cv.best_estimator_cv_.best_estimator_, MockClassifier)
    assert fp_cv.best_estimator_cv_.best_params_ == {"param1": 0, "param2": 0}


def test_fp_estimator_grid_search_verbose(smallest_mols_list, capsys):
    num_mols = len(smallest_mols_list)
    pos_mols = np.ones(num_mols // 2)
    neg_mols = np.zeros(num_mols - len(pos_mols))
    y = np.concatenate((pos_mols, neg_mols))
    fp = AtomPairFingerprint()
    fp_param_grid = {"max_distance": [2, 3, 4]}
    estimator_cv = GridSearchCV(
        estimator=MockClassifier(),
        param_grid={"param1": [0, 1, 2], "param2": [0, 1, 2]},
        scoring="accuracy",
    )
    fp_cv = FingerprintEstimatorGridSearch(fp, fp_param_grid, estimator_cv, verbose=3)
    fp_cv.fit(smallest_mols_list, y)

    output = capsys.readouterr().out
    assert output.startswith("Fitting 3 candidate hyperparameter sets")
    assert "[1/3] START max_distance=2" in output
    assert "[1/3] END max_distance=2; score=0.500; total time=" in output
    assert "[2/3] START max_distance=3" in output
    assert "[2/3] END max_distance=3; score=0.500; total time=" in output
    assert "[3/3] START max_distance=4" in output
    assert "[3/3] END max_distance=4; score=0.500; total time=" in output


def test_best_fp_caching(smallest_mols_list):
    num_mols = len(smallest_mols_list)
    pos_mols = np.ones(num_mols // 2)
    neg_mols = np.zeros(num_mols - len(pos_mols))
    y = np.concatenate((pos_mols, neg_mols))
    fp = AtomPairFingerprint()
    fp_param_grid = {"max_distance": [2, 3, 4]}
    estimator_cv = GridSearchCV(
        estimator=MockClassifier(),
        param_grid={"param1": [0, 1, 2], "param2": [0, 1, 2]},
        scoring="accuracy",
    )
    fp_cv = FingerprintEstimatorGridSearch(
        fp,
        fp_param_grid,
        estimator_cv,
        cache_best_fp_array=True,
        verbose=3,
    )
    fp_cv.fit(smallest_mols_list, y)

    assert fp_cv.best_fp_array_ is not None
    assert isinstance(fp_cv.best_fp_array_, np.ndarray)
    assert fp_cv.best_fp_array_.shape == (num_mols, fp.n_features_out)