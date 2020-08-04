import torch
import math
import random
# import dgl
import torch.nn as nn
import torch.nn.functional as F
from otrans.utils import get_seq_mask, get_dec_seq_mask
from otrans.module import LayerNorm, PositionalEncoding, Embeddings
from otrans.layer import TransformerDecoderLayer, TransformerEncoderLayer
# from otrans.module import G


class TransformerDecoder(nn.Module):
    def __init__(self, output_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6, pos_dropout_rate=0.0, 
                 slf_attn_dropout_rate=0.0, src_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, residual_dropout_rate=0.1,
                 activation='relu', normalize_before=True, concat_after=False, share_embedding=False,):
        super(TransformerDecoder, self).__init__()

        self.normalize_before = normalize_before

        self.embedding = Embeddings(d_model, output_size)
        # self.embedding = G(output_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, pos_dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerDecoderLayer(attention_heads, d_model, linear_units, slf_attn_dropout_rate, src_attn_dropout_rate,
                                    ffn_dropout_rate, residual_dropout_rate, normalize_before=normalize_before, concat_after=concat_after,
                                    activation=activation) for _ in range(num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = LayerNorm(d_model)

        self.output_layer = nn.Linear(d_model, output_size)

        if share_embedding:
            assert self.embedding.weight.size() == self.output_layer.weight.size()
            self.output_layer.weight = self.embedding.weight

        # if share_embedding:
        #     assert self.embedding.ndata['feat'].size() == self.output_layer.weight.size()
        #     self.output_layer.weight = self.embedding.ndata['feat']

    def forward(self, targets, target_length, memory, memory_mask):

        # print(f"targets = {targets}")
        dec_output = self.embedding(targets)
        # print(f"dec_output = {dec_output.size()}")
        dec_output = self.pos_encoding(dec_output)
        # print(f"dec_output2 = {dec_output.size()}")

        dec_mask = get_dec_seq_mask(targets, target_length)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_layer(dec_output)

        return logits, dec_mask

    def recognize(self, preds, memory, memory_mask, last=True):

        dec_output = self.embedding(preds)
        # print(f"preds = {preds.size()}")
        # print(f"memory = {memory}")
        # print(f"dec_output = {dec_output}")
        # print(self.G.ndata['feat'])
        dec_output = self.pos_encoding(dec_output)
        dec_mask = get_seq_mask(preds)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_layer(dec_output)

        log_probs = F.log_softmax(logits[:, -1] if last else logits, dim=-1)

        return log_probs
