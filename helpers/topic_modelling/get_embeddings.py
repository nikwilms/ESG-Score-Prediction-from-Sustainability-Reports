from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

path_to_esg_bert = '/Users/neuefische/repos/ESG-Score-Prediction-from-Sustainability-Reports/models/ESG_BERT'  # Replace this with the actual path or name of ESG-BERT model
tokenizer = BertTokenizer.from_pretrained(path_to_esg_bert, do_lower_case=True)
model = BertModel.from_pretrained(path_to_esg_bert)
model.eval()

def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()[0]

# Example
# embedding = get_embedding("Some financial text from a sustainability report.")

