# Seq2Seq model consisting of encoder and decoder
# support batch processing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, embedding, vocab_size, max_len, hidden_size, dropout_p=0, 
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        self.direction = 1 if bidirectional==False else 2
        self.bidirectional = bidirectional
        
        if rnn_cell.lower() == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        elif rnn_cell.lower() == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
        
        self.variable_lengths = variable_lengths
        self.embedding = embedding
        
    def forward(self, word_inputs, input_lengths, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded, hidden)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return output, hidden

    def init_hidden(self, batch_size):  # pay attention: hidden unit do not obey batch_first
        self.batch_size = batch_size
        hidden = Variable((torch.rand(self.n_layers*self.direction, self.batch_size, self.hidden_size)-0.5)/5)
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


class Decoder(nn.Module):
    def __init__(self, embedding, vocab_size, max_len, hidden_size, dropout_p=0, 
                 n_layers=1, rnn_cell='gru'):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        
        if rnn_cell.lower() == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        elif rnn_cell.lower() == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.emit = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, word_input, last_hidden):
        embedded = self.embedding(word_input)
        output, hidden = self.rnn(embedded, last_hidden)
        output = F.log_softmax(self.emit(output),dim=2)  # B x 1 x N
        
        return output, hidden
