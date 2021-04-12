from transformers import AutoModel, AutoTokenizer
#from model.transformers.modeling_bert import BertModel
from utility.bert_tokenizer import bert_tokenizer
import torch
from torch import nn
import torch.nn.functional as F

if __name__ == "__main__":

    model = AutoModel.from_pretrained("bert-base-multilingual-uncased", output_hidden_states=True)

    # These parameters automatically learn to combine representations from all
    # Bert layers in a gradient-based way
    layer_scores = nn.Parameter(torch.zeros(12, 1, 1, 1), requires_grad=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

    example = {"input": "mobilnetverket jeg har er ikke akkurat imponerende".split()}

    to_scatter, bert_input = bert_tokenizer(example, tokenizer, "bert-base-multilingual-uncased")

    bert_input = torch.LongTensor(bert_input).unsqueeze(0)
    attention_mask = torch.ones_like(bert_input)
    encoding = model(bert_input, attention_mask=attention_mask)
    # encoded.shape =  (bert_depth, batch_size, length, dim) = (12, B, L, 768)
    encoded = torch.stack(encoding["hidden_states"][1:], dim=0)
    scores = F.softmax(layer_scores, dim=0)
    encoded = (scores * encoded).sum(0) # shape: (batch_size, length, dim)
