from sklearn.feature_extraction.text import TfidfVectorizer
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import numpy as np
from scipy.sparse import csr_matrix
from .normalization_methods import *

class TfidfRelu(BaseFeatureExtraction):
    name = "tfidfrelu"

    def __init__(self, *args, ngram_max=1, stop_words="english", **kwargs):
        """Initialize tfidf class."""
        super().__init__(*args, **kwargs)
        self.ngram_max = ngram_max
        self.stop_words = stop_words
        if stop_words is None or stop_words.lower() == "none":
            sklearn_stop_words = None
        else:
            sklearn_stop_words = self.stop_words
        self._model = TfidfVectorizer(
            ngram_range=(1, ngram_max), stop_words=sklearn_stop_words
        )

    def fit(self, texts):
        self._model.fit(texts)

    def transform(self, texts):
        print("%%%%%%%%%%%%%%% RELU %%%%%%%%%%%%%")
        X = self._model.transform(texts).tocsr()
        X = relu(X)
        print(X)
        return X