from sklearn.feature_extraction.text import TfidfVectorizer
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import numpy as np
from scipy.sparse import csr_matrix
from .normalization_methods import *

class TfidfAddAbsMin(BaseFeatureExtraction):
    name = "tfidfaddabsmin"

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
        """Fit the TF-IDF model."""
        self._model.fit(texts)

    def transform(self, texts):
        """Transform texts using the fitted TF-IDF model and normalize by adding the absolute value of the lowest number."""
        print("%%%%%%%%%%%%% ABS MIN %%%%%%%%%%%%%%")
        X = self._model.transform(texts).tocsr()
        X = absmin(X)
        print(X)
        return X