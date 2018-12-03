'''
WARNING: This code is full of bugs. Can you squash them all?
We've created a super-awesome sentiment classification tool
that recognizes whether a movie review is good or bad.
However, it does not work as expected... Why?
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This seems like a rough implementation of
# [Yoon Kim, 2014, Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
# ... but is it really so?

# NOTE: you do not need to implement everything (for example, every hyperparameter)
# as proposed in the paper to make the code work.
class CnnClassifier(nn.Module):
    def __init__(self, vocab_size,
                 vocab_embed_size=50,
                 filter_sizes=[2, 3, 4, 5],
                 out_dim=300,
                 num_classes=2,
                 dropout=0.99,
                 pad_index=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_embed_size = vocab_embed_size
        self.pad_index = pad_index

        # we will use multiple width convolution filters to capture character patterns for a sentiment
        self.filter_sizes = sorted(filter_sizes)
        # we will concat the max-pooled outputs of the convolutions
        self.num_filters = out_dim // len(filter_sizes)
        # this is the concatenated hidden dimension
        self.out_dim = out_dim
        # since this is a binary classification problem, number of classes is 2
        self.num_classes = num_classes

        assert not self.out_dim % self.num_filters

        # character embedding matrix
        self.embedding = nn.Embedding(
            self.vocab_size, self.vocab_embed_size, 
            padding_idx=self.pad_index)
        self.embedding_dropout = nn.Dropout(p=dropout)
        
        self.pool = nn.AdaptiveMaxPool1d(1) # we will max pool the output of convolution filters of each width
        self.convs = [nn.Conv1d(self.vocab_embed_size, self.num_filters, width).to(device) 
                 for width in filter_sizes] # convolution for filters of varying width
        self.fc = nn.Linear(self.out_dim, self.num_classes) # final layer to convert deep features to binary scores
        self.fc_dropout = nn.Dropout(p=dropout)
        
        # weight initialization
        for p in self.parameters():
            p.data.fill_(0)

    def forward(self, x, real_len=None):
        # lengths of the real inputs.
        # the data would be padded like [word, world, ..., last_word, pad, pad, pad ...]
        # so we can only calculate the number of non-padding characters
        # to know the real lengths of sequences, not zero-padded...
        if real_len is None:
            real_len = torch.sum(x.ne(self.pad_index), dim=0)
        x = self.embedding_dropout(self.embedding(x))
        
        # exception handling, in case that the number of characters in the sentence
        # is smaller than the convolution filter width.
        # pad with zero, so that error does not occur.
        if x.size(1) < max(self.filter_sizes):
            x = torch.cat([x, torch.zeros(
                x.size(0),
                max(self.filter_sizes) - x.size(1),
                x.size(2)).to(device)], dim=1)
        
        # we can avoid unnecessary calculations by excluding the all-zero part
        # from the input.
        L = max(real_len.max().int(), max(self.filter_sizes))
        x = x[:, :L, :] # [B, L', D]

        # reshape the input to fit to the input shape for convolutions
        x = x.contiguous().view(-1, x.size(2), x.size(1))   # [B, D, L']

        conv_outputs = []   # gather convolution outputs here
        for conv in self.convs:
            conv_outputs += [self.pool(F.relu(conv(x))).squeeze(-1)]
        hidden = torch.cat(conv_outputs, dim=-1)  # concatenate to get [B, D']
        return self.fc_dropout(self.fc(hidden)) # [B, 2]
