from torch.cuda.amp import autocast
from asreview.models.feature_extraction.base import BaseFeatureExtraction

class Fasttext(BaseFeatureExtraction):

    name = "fasttext"

    def __init__(self):

        super().__init__()

    def fit(self,  texts):
        """Fit the model to the data."""
        return None

    def transform(self, texts):
       # Skipgram model :
        model = fasttext.train_unsupervised(texts, model='skipgram')

        # # or, cbow model :
        # model = fasttext.train_unsupervised(texts, model='cbow')
        print(model.words[1])
        return model
