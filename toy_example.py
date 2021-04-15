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


labels = ["<sow>", "<eow>", "O", "holder1", "holder2", "target1", "target2", "exp1", "exp2"]
label2idx = dict([(label, i) for i, label in enumerate(labels)])


def gold_to_tensor(gold_spans, label2idx):
    # remove <sow> tokens
    spans = [i[1:] for i in gold_spans]
    # convert_to_idxs
    idx_spans = [torch.LongTensor([label2idx[l] for l in span]) for span in spans]
    # convert to torch.LongTensor
    gold_tensor = torch.cat(idx_spans)
    return gold_tensor

class SpanDecoder(nn.Module):
    def __init__(self, args, hidden_dim, label2idx, label_emb_dim=50):
        super(SpanDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.label2idx = label2idx
        self.idx2label = dict([(i, l) for l, i in label2idx.items()])
        self.label_emb_dim = label_emb_dim
        #
        self.label_embedding = nn.Embedding(len(label2idx),
                                            self.label_emb_dim)
        self.lstm = nn.LSTM(768 + self.label_emb_dim * 2, 50)
        self.linear = nn.Linear(50, len(label2idx))
        #
        self.max_word_label_length = 5
        self.start_tok = "<sow>"
        self.end_tok = "<eow>"

    def token_lstm(self, avg_emb_t_minus_1, current_encoder_output, gold_length=None, train=True):
        """ Initialize LSTM input with start embedding
            lstm_input [1 x 768 + label_emb_dim * 2]
        LSTM input is (avg emb from t-1, prev. output emb t, encoder output t)
        """
        token_outputs = []
        token_preds = ["<sow>"]
        if gold_length is not None:
            token_range = range(gold_length)
        else:
            token_range = range(self.max_word_label_length)
        for j in token_range:
            prev_output_idx = torch.LongTensor([label2idx[token_preds[-1]]])
            previous_output_emb = self.label_embedding(prev_output_idx)
            lstm_input = torch.cat((avg_emb_t_minus_1, previous_output_emb, current_encoder_output), dim=1)
            # lstm_input needs 3 dims: [1 x 1 x 768 + label_emb_dim * 2]
            lstm_input = lstm_input.unsqueeze(0)
            out, hidden = self.lstm(lstm_input)
            out = self.linear(out).squeeze(0)
            _, p = out.max(dim=1)
            pred_idx = int(p[0])
            pred_label = self.idx2label[pred_idx]
            token_outputs.append(out)
            token_preds.append(pred_label)
            # During inference, stop if you reach <eow>
            if pred_label == self.end_tok and gold_length is None and train == False:
                break
        token_outputs = torch.cat(token_outputs, dim=0)
        return token_outputs, token_preds

    def avg_label_embedding(self, list_of_labels):
        lp = torch.LongTensor([self.label2idx[l] for l in list_of_labels])
        embedded = self.label_embedding(lp)
        avg_emb = embedded.mean(dim=0).unsqueeze(0)  # [1 x label_emb_dim]
        return avg_emb

    def forward(self, encoder_output, gold_labels=None, train=True):

        batch_size, seq_length, encoder_dim = encoder_output.shape
        if gold_labels is not None:
            # The gold_labels all start with <sow>, which is added automatically, so we need to reduce the lengths by 1
            lengths = [len(i) - 1 for i in gold_labels]
        else:
            lengths = [self.max_word_label_length] * seq_length

        start_idx = self.label2idx[self.start_tok]
        end_idx = self.label2idx[self.end_tok]
        start_emb = self.label_embedding(torch.LongTensor([start_idx]))
        #
        decoder_outputs = []
        decoder_predictions = []

        for i in range(seq_length):
            current_encoder_output = encoder_output[:, i, :]
            if i == 0:
                prev_emb = start_emb
                token_outputs, token_predictions = self.token_lstm(prev_emb,
                                                                   current_encoder_output,
                                                                   lengths[i])
            else:
                prev_emb = self.avg_label_embedding(decoder_predictions[-1])
                token_outputs, token_predictions = self.token_lstm(prev_emb,
                                                                   current_encoder_output,
                                                                   lengths[i])
            decoder_outputs.append(token_outputs)
            decoder_predictions.append(token_predictions)
        decoder_outputs = torch.cat(decoder_outputs)
        return decoder_outputs, decoder_predictions


class RelationDecoder(nn.Module):
    def __init__(self, args, hidden_dim, num_classes):
        super(RelationDecoder, self).__init__()
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


def encode(sent, encoder):
    to_scatter, idxs = bert_tokenizer(sent, tokenizer, "bert-base-multilingual-uncased")

    # Need at least one batch, so unsqueeze at dim=0
    tokens = torch.LongTensor(idxs).unsqueeze(0)
    attention_mask = torch.ones_like(tokens)
    bert_input = (tokens, attention_mask)
    to_scatter = torch.LongTensor(to_scatter).unsqueeze(0)
    n_words = len(sent["input"])

    encoder_output, decoder_input = encoder.forward(bert_input, to_scatter, n_words)
    return encoder_output, decoder_input

def decode(sent, encoder_output, decoder, return_both=True):
    holder_idxs = sent["holders"]
    target_idxs = sent["targets"]
    exp_idxs = sent["exps"]

    num_holder = len(holder_idxs)
    num_target = len(target_idxs)
    num_exp = len(exp_idxs)
    num_classes = 3
    pred_tensor = decoder.get_relations(encoder_output,
                                        holder_idxs,
                                        target_idxs,
                                        exp_idxs)

    if return_both:
        gold_tensor = torch.LongTensor(sent["relations"])
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

    example1 = {"input": "<null> I really loved the great views !".split(),
                "output": [(["I"], ["the great views"],["really loved"],"POS"),
                           (["I"], ["views"], ["great"], "POS")
                           ],
                "spans": [["<sow>", "O", "<eow>"],
                          ["<sow>", "holder1", "holder2", "<eow>"],
                          ["<sow>", "exp1", "<eow>"],
                          ["<sow>", "exp1", "<eow>"],
                          ["<sow>", "target1", "<eow>"],
                          ["<sow>", "target1", "exp2", "<eow>"],
                          ["<sow>", "target1", "target2", "<eow>"],
                          ["<sow>", "O", "<eow>"]],
                "holders": [[0], [1]],
                "targets": [[0], [4, 5, 6], [6]],
                "exps": [[1, 2], [4]],
                "relations":  [[[0, 0],
                                [0, 0],
                                [0, 0]],
                               [[0, 0],
                                [2, 0],
                                [2, 0]]]
                }
    example2 = {"input": "<null> I like Peppes more than Dominos !".split(),
                "output": [(["I"], ["Peppes"], ["like", "more than"], "POS"),
                           (["I"], ["Dominos"], ["like", "more than"], "NEG")
                           ],
                "spans": [["<sow>", "O", "<eow>"],
                          ["<sow>", "holder1", "holder2", "<eow>"],
                          ["<sow>", "exp1", "exp2", "<eow>"],
                          ["<sow>", "target1", "<eow>"],
                          ["<sow>", "exp1", "exp2", "<eow>"],
                          ["<sow>", "exp1", "exp2", "<eow>"],
                          ["<sow>", "target2", "<eow>"],
                          ["<sow>", "O", "<eow>"]],
                "holders": [[0], [1]],
                "targets": [[0], [3], [6]],
                "exps": [[2, 4, 5]],
                "relations": [[[0],
                               [0],
                               [0]],
                              [[0],
                               [2],
                               [1]]]
                }

    example3 = {"input": "<null> The TV was pretty cool".split(),
                "output": [(["<null>"], ["The TV"], ["pretty cool"], "POS")],
                "spans": [["<sow>", "holder1", "<eow>"],
                          ["<sow>", "target1", "<eow>"],
                          ["<sow>", "target1", "<eow>"],
                          ["<sow>", "O", "<eow>"],
                          ["<sow>", "exp1", "<eow>"],
                          ["<sow>", "exp1", "<eow>"]
                          ],
                "holders": [[0]],
                "targets": [[0], [1, 2]],
                "exps": [[4, 5]],
                "relations": [[[0],
                               [2]]]
                }

    example4 = {"input": "<null> Definitely pretty cool , but could be bigger".split(),
                "output": [(["<null>"], ["<null>"], ["pretty cool"], "POS")],
                "spans": [["<sow>", "holder1", "holder2", "target1", "target2", "<eow>"],
                          ["<sow>", "O", "<eow>"],
                          ["<sow>", "exp1", "<eow>"],
                          ["<sow>", "exp1", "<eow>"],
                          ["<sow>", "O", "<eow>"],
                          ["<sow>", "O", "<eow>"],
                          ["<sow>", "exp2", "<eow>"],
                          ["<sow>", "exp2", "<eow>"],
                          ["<sow>", "exp2", "<eow>"],
                          ],
                "holders": [[0]],
                "targets": [[0]],
                "exps": [[1, 2, 3], [6, 7, 8]],
                "relations": [[[2], [1]]]
                }

    example5 = {"input": "<null> The screen is not cool".split(),
                "output": [(["<null>"], ["The screen"], ["not cool"], "NEG")],
                "spans": [["<sow>", "holder1", "<eow>"],
                          ["<sow>", "target1", "<eow>"],
                          ["<sow>", "target1", "<eow>"],
                          ["<sow>", "O", "<eow>"],
                          ["<sow>", "exp1", "<eow>"],
                          ["<sow>", "exp1", "<eow>"],
                          ],
                "holders": [[0]],
                "targets": [[0], [1, 2]],
                "exps": [[4, 5]],
                "relations": [[[0],
                               [1]]]
                }

    example6 = {"input": "<null> I like the views".split(),
                "output": [(["I"], ["the views"], ["like"], "POS")],
                "holders": [[0], [1]],
                "targets": [[0], [3, 4]],
                "exps": [[2]],
                "relations": [[[0],
                               [0]],
                              [[0],
                               [2]]]
                }

    example7 = {"input": "<null> great TV".split(),
                "output": [(["null"], ["TV"], ["great"], "POS")],
                "holders": [[0]],
                "targets": [[0], [2]],
                "exps": [[1]],
                "relations": [[[0],
                               [2]]]
                }




    # Number of classes == 3 (Positive, Negative, None)
    encoder = Encoder(args, None)
    rel_decoder = RelationDecoder(args, hidden_dim=50, num_classes=3)
    span_decoder = SpanDecoder(args, 50, label2idx)

    span_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(span_decoder.parameters()))

    rel_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(rel_decoder.parameters()))


    # Train span extraction
    print("Training span prediction")
    for i in range(50):
        full_loss = 0.0
        for sent in [example1, example2, example3, example4, example5]:
            span_optimizer.zero_grad()
            encoder_output, decoder_input = encode(sent, encoder)
            gold_labels = sent["spans"]
            preds, pred_labels = span_decoder(encoder_output, gold_labels)
            y = gold_to_tensor(gold_labels, label2idx)
            loss = F.cross_entropy(preds, y)
            full_loss += loss
            #loss.backward()
            #span_optimizer.step()

            """
        print("Loss: {0:.3f}".format(full_loss.data))
        full_loss.backward()
        span_optimizer.step()

    # Train relation prediction
    print("Training relation prediction")
    for i in range(15):

        full_loss = 0.0

        for sent in [example1, example2, example3, example4, example5]:
            rel_optimizer.zero_grad()

            encoder_output, decoder_input = encode(sent, encoder)
            """
            pred_tensor, gold_tensor = decode(sent, encoder_output, rel_decoder)

            loss = F.cross_entropy(pred_tensor, gold_tensor)
            full_loss += loss

        print("Loss: {0:.3f}".format(full_loss.data))
        full_loss.backward()
        rel_optimizer.step()

