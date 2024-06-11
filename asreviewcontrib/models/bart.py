from asreview.models.feature_extraction.base import BaseFeatureExtraction
from transformers import BartTokenizer, BartModel
from torch.cuda.amp import autocast

class Bart(BaseFeatureExtraction):

    name = "bart"

    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model = BartModel.from_pretrained('facebook/bart-base')
        super().__init__()

    def fit(self,  texts):
        """Fit the model to the data."""
        return None


    def transform(self, texts):
        conv_list = texts.tolist()
        inputs = self.tokenizer(conv_list, return_tensors="pt", padding=True, truncation=True,max_length=512)
        
        with autocast():
            outputs = self.model(**inputs)

        return outputs