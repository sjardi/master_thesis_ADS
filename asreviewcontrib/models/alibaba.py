from asreview.models.feature_extraction.base import BaseFeatureExtraction
from transformers import pipeline
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class Alibaba(BaseFeatureExtraction):

    name = "alibaba"

    def __init__(self):

        # # (Optionally) normalize embeddings
        # embeddings = F.normalize(embeddings, p=2, dim=1)
        # scores = (embeddings[:1] @ embeddings[1:].T) * 100
        # print(scores.tolist())

        super().__init__()

    def fit(self,  texts):
        """Fit the model to the data."""
        return None

    def transform(self, texts):
        print(type(texts.tolist()))
        print(texts.tolist())

        input_texts = [
            "what is the capital of China?",
            "how to implement quick sort in python?",
            "Beijing",
            "sorting algorithms"
        ]
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        _model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        batch_dict = tokenizer(texts.tolist(), max_length=8192,
                               padding=True, truncation=True, return_tensors='pt')
        outputs = _model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]
        return embeddings


# embeddings = Alibaba.transform("test")
# print(embeddings)
# Requires transformers>=4.36.0
