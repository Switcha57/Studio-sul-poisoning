import lightgbm
import numpy as np
#from ember import PEFeatureExtractor
from .custom_features import PEFeatureExtractor
from secml.array import CArray
from secml.ml.classifiers import CClassifier


class CustomClassifierEmber(CClassifier):
    """
	The wrapper for the EMBER GBDT, by Anderson et al. https://arxiv.org/abs/1804.04637
	"""

    def __init__(self, tree_path: str = None):
        """
		Create the EMBER tree.

		Parameters
		----------
		tree_path : str
			path to the tree parameters
		"""
        super(CClassifier, self).__init__()
        self._lightgbm_model = tree_path  # self._load_tree(tree_path)
    
    def extract_features(self, x: CArray) -> CArray:
        """
		Extract EMBER features

		Parameters
		----------
		x : CArray
			program sample
		Returns
		-------
		CArray
			EMBER features
		"""
        extractor = PEFeatureExtractor(2, print_feature_warning=False)
        x = x.atleast_2d()
        size = x.shape[0]
        features = []
        for i in range(size):
            x_i = x[i, :]
            x_bytes = bytes(x_i.astype(np.int).tolist()[0])
            features.append(np.array(extractor.feature_vector(x_bytes), dtype=np.float32))
        features = CArray(features)
        return features

    def _backward(self, w):
        pass

    def _fit(self, x, y):
        raise NotImplementedError("Fit is not implemented.")

    def _load_tree(self, tree_path):
        booster = lightgbm.Booster(model_file=tree_path)
        self._classes = 2
        self._n_features = booster.num_feature()
        return booster

    def _forward(self, x):
        x = x.atleast_2d()
        scores = self._lightgbm_model.predict(x.tondarray(), verbose=0)
        # scores=np.argmax(scores,axis=-1)
        confidence = scores  # [[1 - c, c] for c in scores]
        confidence = CArray(confidence)
        return confidence


    def predict(self, x, return_decision_function=False):
        scores = self.decision_function(x, y=None)

        # Checking if the score is higher than ember model threshold
        labels = (scores > 0.82).astype(int)

        label = labels.argmax(axis=1).ravel()

        return (label, scores) if return_decision_function is True else labels


