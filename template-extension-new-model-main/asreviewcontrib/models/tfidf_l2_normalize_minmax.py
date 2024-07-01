from sklearn.feature_extraction.text import TfidfVectorizer
from asreview.models.feature_extraction.base import BaseFeatureExtraction
from .normalization_methods import *


class Tfidf_l2_normalize_minmax(BaseFeatureExtraction):
    name = "tfidf_l2_normalize_minmax"

    def __init__(self, *args, ngram_max=1, stop_words="english", **kwargs):
        """Initialize tfidf class."""
        super().__init__(*args, **kwargs)
        self.ngram_max = ngram_max
        self.stop_words = stop_words
        self.name = "tfidf_l2_normalize_minmax"
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
        print("%%%%%%%%%%%% TF_IDF l2_normalize_minmax %%%%%%%%%%%%%%")
        X = self._model.transform(texts).tocsr()
        X = l2_normalize_minmax(X)
        print("Adding embeddings")
        if X.shape[0] > 0:
            add_embeddings(X, self.name)
        else:
            print("############ Warning: X is empty, skipping add_embeddings ##########")
        print(X)
        return X
