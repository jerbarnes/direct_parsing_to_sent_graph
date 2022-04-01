#!/usr/bin/env python3
# coding=utf-8

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert import BertModel
from transformers.models.xlm_roberta import XLMRobertaModel
from transformers import RobertaConfig
from model.module.char_embedding import CharEmbedding


class WordDropout(nn.Dropout):
    def forward(self, input_tensor):
        if self.p == 0:
            return input_tensor

        ones = input_tensor.new_ones(input_tensor.shape[:-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)

        return dropout_mask.unsqueeze(-1) * input_tensor


class Encoder(nn.Module):
    def __init__(self, args, dataset):
        super(Encoder, self).__init__()

        self.dim = args.hidden_size
        self.n_layers = args.n_encoder_layers
        self.width_factor = args.query_length

        if "roberta" in args.encoder.lower():
            config = RobertaConfig.from_pretrained(args.encoder)
            config.update({"gradient_checkpointing": True})
            self.bert = XLMRobertaModel(config, add_pooling_layer=False)
            if args.encoder_freeze_embedding:
                self.bert.embeddings.requires_grad_(False)
                self.bert.embeddings.LayerNorm.requires_grad_(True)
        else:
            self.bert = BertModel.from_pretrained(args.encoder, gradient_checkpointing=True, add_pooling_layer=False)
        
        if args.freeze_bert:
            self.bert.requires_grad_(False)

        self.use_char_embedding = args.char_embedding
        if self.use_char_embedding:
            self.form_char_embedding = CharEmbedding(dataset.char_form_vocab_size, args.char_embedding_size, self.dim)
            self.word_dropout = WordDropout(args.dropout_word)

        self.post_layer_norm = nn.LayerNorm(self.dim)
        self.subword_attention = nn.Linear(self.dim, 1)

        if self.width_factor > 1:
            self.query_generator = nn.Linear(self.dim, self.dim * self.width_factor)
        else:
            self.query_generator = nn.Identity()

        self.encoded_layer_norm = nn.LayerNorm(self.dim)
        self.scores = nn.Parameter(torch.zeros(self.n_layers, 1, 1, 1), requires_grad=True)

    def forward(self, bert_input, form_chars, to_scatter, n_words):
        tokens, mask = bert_input
        batch_size = tokens.size(0)

        encoded = self.bert(tokens, attention_mask=mask, output_hidden_states=True).hidden_states[1:]
        encoded = torch.stack(encoded, dim=0)  # shape: (12, B, T, H)
        encoded = self.encoded_layer_norm(encoded)

        if self.training:
            time_len = encoded.size(2)
            scores = self.scores.expand(-1, batch_size, time_len, -1)
            dropout = torch.empty(self.n_layers, batch_size, 1, 1, dtype=torch.bool, device=self.scores.device)
            dropout.bernoulli_(0.1)
            scores = scores.masked_fill(dropout, float("-inf"))
        else:
            scores = self.scores

        scores = F.softmax(scores, dim=0)
        encoded = (scores * encoded).sum(0)  # shape: (B, T, H)
        encoded = encoded.masked_fill(mask.unsqueeze(-1) == 0, 0.0)[:, 1:-1, :]

        subword_attention = self.subword_attention(encoded).sigmoid()  # shape: (B, T, 3)
        encoded = encoded * subword_attention

        to_scatter = to_scatter.unsqueeze(-1).expand(-1, -1, self.dim)
        encoder_output = torch.zeros(encoded.size(0), n_words + 1, self.dim, device=encoded.device)
        encoder_output.scatter_add_(dim=1, index=to_scatter, src=encoded)  # shape: (B, n_words + 1, H)
        encoder_output = encoder_output[:, :-1, :]
        encoder_output = self.post_layer_norm(encoder_output)
        if self.use_char_embedding:
            form_char_embedding = self.form_char_embedding(form_chars[0], form_chars[1], form_chars[2])
            encoder_output = self.word_dropout(encoder_output) + form_char_embedding

        decoder_input = self.query_generator(encoder_output)
        decoder_input = decoder_input.view(batch_size, -1, self.width_factor, self.dim).flatten(1, 2)  # shape: (B, T*Q, D)

        return encoder_output, decoder_input
