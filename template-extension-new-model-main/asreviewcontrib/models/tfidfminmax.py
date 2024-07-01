from sklearn.feature_extraction.text import TfidfVectorizer
from asreview.models.feature_extraction.base import BaseFeatureExtraction
import numpy as np
from scipy.sparse import csr_matrix
from .normalization_methods import *


class TfidfMinMax(BaseFeatureExtraction):
    name = "tfidfminmax"

    def __init__(self, *args, ngram_max=1, stop_words="english", new_min=0, new_max=1, **kwargs):
        """Initialize tfidf class."""
        super().__init__(*args, **kwargs)
        self.ngram_max = ngram_max
        self.stop_words = stop_words
        self.new_min = new_min
        self.new_max = new_max
        self.name = "tfidf_minmax"
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
        print("%%%%%%%%%%%%% MIN-MAX with TF-IDF %%%%%%%%%%%%%")
        X = self._model.transform(texts).tocsr()
        X = minmax(X)
        print("Adding embeddings")
        if X.shape[0] > 0:
            add_embeddings(X, self.name)
        else:
            print("############ Warning: X is empty, skipping add_embeddings ##########")
        
        print(X)
        return X
