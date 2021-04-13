from transformers import AutoModel, AutoTokenizer
#from model.transformers.modeling_bert import BertModel
#from model.transformers.base import Decoder
from utils.bert_tokenizer import bert_tokenizer
from utils.utils import create_padding_mask
import torch
from torch import nn
import torch.nn.functional as F
import math
import argparse

class Decoder(nn.Module):
    def __init__(self, args, hidden_dim, num_classes):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rel_pred = args.rel_pred

        self.holder_lstm = nn.LSTM(args.hidden_size, self.hidden_dim, bidirectional=True)
        self.target_lstm = nn.LSTM(args.hidden_size, self.hidden_dim, bidirectional=True)
        self.exp_lstm = nn.LSTM(args.hidden_size, self.hidden_dim, bidirectional=True)

        if self.rel_pred == "biaffine":
            self.holder_exp_T = nn.Parameter(torch.Tensor(self.hidden_dim*2,
                                                          self.hidden_dim*2,
                                                          self.hidden_dim*2))
            self.target_exp_T = nn.Parameter(torch.Tensor(self.hidden_dim*2,
                                                          self.hidden_dim*2,
                                                          self.hidden_dim*2))
            self.classification_T = nn.Parameter(torch.Tensor(self.hidden_dim*2,
                                                              self.num_classes,
                                                              self.hidden_dim*2))
        elif self.rel_pred == "feedforward":
            self.ff = nn.Linear(hidden_dim*6, num_classes)
        else:
            print("{} not implemented".format(self.rel_pred))

    def lstm_pool(self, entity_encoding, lstm):
        # entity_encoding = (batch_size, sequence_length, hidden_dim)
        output, _ = lstm(entity_encoding)  # output (bsize, seq_l, lstm_dim*2)
        max_out, _ = output.max(1)  # max_out = (bsize, lstm_dim*2)
        return max_out.squeeze(0)

    def get_relations(self, encoder_output, holder_idxs, target_idxs, exp_idxs):
        holder_encodings = [encoder_output[:,hidx,:] for hidx in holder_idxs]
        target_encodings = [encoder_output[:,tidx,:] for tidx in target_idxs]
        exp_encodings = [encoder_output[:,eidx,:] for eidx in exp_idxs]

        holder_reps = [self.lstm_pool(henc, self.holder_lstm) for henc in holder_encodings]
        target_reps = [self.lstm_pool(tenc, self.target_lstm) for tenc in target_encodings]
        exp_reps = [self.lstm_pool(eenc, self.exp_lstm) for eenc in exp_encodings]

        # get relationship scores for all holder, target, exp permutations
        # pred_tensor = (num_holders, num_targets, num_exp, polarity_classes)
        pred_tensor = torch.zeros(len(holder_reps),
                                  len(target_reps),
                                  len(exp_reps),
                                  self.num_classes)

        for i, he in enumerate(holder_reps):
            for j, te in enumerate(target_reps):
                for k, ee in enumerate(exp_reps):
                    if self.rel_pred == "biaffine":
                        hep = torch.matmul(torch.matmul(he, self.holder_exp_T), ee)
                        tep = torch.matmul(torch.matmul(te, self.target_exp_T), ee)
                        pred = torch.matmul(tep, torch.matmul(self.classification_T, hep))
                        #pred = F.softmax(pred)
                    elif self.rel_pred == "feedforward":
                        conc = torch.cat((he, te, ee))
                        pred = self.ff(conc)
                        #pred = F.softmax(pred)
                    pred_tensor[i][j][k] = pred

        return pred_tensor

class QueryGenerator(nn.Module):
    def __init__(self, dim, width_factor):
        super(QueryGenerator, self).__init__()

        weight = torch.Tensor(width_factor * dim, dim)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight.t().repeat(1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, width_factor * dim))
        self.width_factor = width_factor

    def forward(self, encoder_output):
        batch_size, seq_len, dim = encoder_output.shape

        # shape: (B, T, Q*D)
        queries = encoder_output.matmul(self.weight) + self.bias
        queries = torch.tanh(queries)  # shape: (B, T, Q*D)
        queries = queries.view(batch_size, seq_len, self.width_factor, dim).flatten(1, 2)  # shape: (B, T*Q, D)
        return queries

class Encoder(nn.Module):
    def __init__(self, args, dataset):
        super(Encoder, self).__init__()

        self.dim = args.hidden_size
        self.n_layers = args.n_encoder_layers
        self.width_factor = args.query_length

        self.bert = AutoModel.from_pretrained(args.encoder, output_hidden_states=True)


        self.query_generator = QueryGenerator(self.dim, self.width_factor)
        self.encoded_layer_norm = nn.LayerNorm(self.dim)
        self.scores = nn.Parameter(torch.zeros(self.n_layers, 1, 1, 1), requires_grad=True)

    def forward(self, bert_input, to_scatter, n_words):
        tokens, mask = bert_input
        batch_size = tokens.size(0)
        encoding = self.bert(tokens, attention_mask=mask)

            # encoded.shape =  (bert_depth, batch_size, length, dim) = (12, B, L, 768)
        encoded = torch.stack(encoding["hidden_states"][1:], dim=0)

        if self.training:
            time_len = encoded.size(2)
            scores = self.scores.expand(-1, batch_size, time_len, -1)
            dropout = torch.empty(self.n_layers, batch_size, 1, 1, dtype=torch.bool, device=self.scores.device)
            dropout.bernoulli_(0.1)
            scores = scores.masked_fill(dropout, float("-inf"))
        else:
            scores = self.scores

        scores = F.softmax(scores, dim=0)

        encoded = (scores * encoded).sum(0) # shape: (batch_size, length, dim)

        to_scatter = to_scatter.unsqueeze(-1).expand(-1, -1, self.dim)
        encoder_output = torch.zeros(encoded.size(0), n_words + 1, self.dim, device=encoded.device)
        encoder_output.scatter_add_(dim=1, index=to_scatter, src=encoded[:, 1:-1, :])  # shape: (batch_size, n_words + 1, dim)
        encoder_output = encoder_output[:, :-1, :] # shape: (B, n_words, dim)

        decoder_input = self.query_generator(encoder_output)
        return encoder_output, decoder_input


def predict_relations(sent, encoder, decoder, return_both=True):
    to_scatter, idxs = bert_tokenizer(sent, tokenizer, "bert-base-multilingual-uncased")

    # Need at least one batch, so unsqueeze at dim=0
    tokens = torch.LongTensor(idxs).unsqueeze(0)
    attention_mask = torch.ones_like(tokens)
    bert_input = (tokens, attention_mask)
    to_scatter = torch.LongTensor(to_scatter).unsqueeze(0)
    n_words = len(sent["input"])

    holder_idxs = sent["holders"]
    target_idxs = sent["targets"]
    exp_idxs = sent["exps"]

    gold_tensor = torch.LongTensor(sent["relations"])

    num_holder = len(holder_idxs)
    num_target = len(target_idxs)
    num_exp = len(exp_idxs)
    num_classes = 3

    encoder_output, decoder_input = encoder.forward(bert_input, to_scatter, n_words)
    pred_tensor = decoder.get_relations(encoder_output,
                                        holder_idxs,
                                        target_idxs,
                                        exp_idxs)

    if return_both:
        pred_tensor = pred_tensor.reshape((num_holder * num_target * num_exp, num_classes))
        gold_tensor = gold_tensor.reshape(num_holder * num_target * num_exp)
        return pred_tensor, gold_tensor
    else:
        return pred_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="bert-base-multilingual-uncased")
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--hidden_size_ff", default=4*768, type=int)
    parser.add_argument("--n_encoder_layers", default=12, type=int)
    parser.add_argument("--n_layers", default=12, type=int)
    parser.add_argument("--n_attention_heads", default=6, type=int)
    parser.add_argument("--dropout_transformer_attention", default=.1, type=float)
    parser.add_argument("--dropout_transformer", default=.1, type=float)
    parser.add_argument("--query_length", default=2, type=int)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--pre_norm", default=True)
    parser.add_argument("--rel_pred",
                        default="feedforward",
                        help="biaffine or feedforward")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    example1 = {"input": "I really loved the great views !".split(),
                "output": [(["I"], ["the great views"],["really loved"],"POS"),
                           (["I"], ["views"], ["great"], "POS")
                           ],
                "holders": [[0]],
                "targets": [[3, 4, 5], [5]],
                "exps": [[1, 2], [4]],
                "relations": [[[2, 0], [2, 0]]]
                }
    example2 = {"input": "I like Peppes more than Dominos !".split(),
                "output": [(["I"], ["Peppes"], ["like", "more than"], "POS"),
                           (["I"], ["Dominos"], ["like", "more than"], "NEG")
                           ],
                "holders": [[0]],
                "targets": [[2], [5]],
                "exps": [[1, 3, 4]],
                "relations": [[[2], [1]]]
                }

    example3 = {"input": "<null> The TV was pretty cool".split(),
                "output": [(["<null>"], ["The TV"], ["pretty cool"], "POS")],
                "holders": [[0]],
                "targets": [[1, 2]],
                "exps": [[4, 5]],
                "relations": [[[2]]]
                }

    example4 = {"input": "<null> Definitely pretty cool , but could be bigger".split(),
                "output": [(["<null>"], ["<null>"], ["pretty cool"], "POS")],
                "holders": [[0]],
                "targets": [[0]],
                "exps": [[1, 2, 3], [6, 7, 8]],
                "relations": [[[2], [1]]]
                }




    # Number of classes == 3 (Positive, Negative, None)
    encoder = Encoder(args, None)
    decoder = Decoder(args, hidden_dim=50, num_classes=3)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))


    for i in range(15):

        for sent in [example1, example2, example3, example4]:
            optimizer.zero_grad()

            pred_tensor, gold_tensor = predict_relations(sent, encoder, decoder)

            loss = F.cross_entropy(pred_tensor, gold_tensor)
            print("Loss: {0:.3f}".format(loss.data))
            loss.backward()
            optimizer.step()

