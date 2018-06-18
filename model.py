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

# batch_first
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
        
        self.embedding = embedding
        self.emit = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, word_input, last_hidden):
        embedded = self.embedding(word_input)
        output, hidden = self.rnn(embedded, last_hidden)
        output = F.log_softmax(self.emit(output),dim=2)  # B x 1 x N
        
        return output, hidden

    
# currently only support general_batch for batch processing
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general' or self.method == 'general_batch':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [1,B,N]
        # encoder_outputs: [S,B,N]
        
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        self.method = 'general_batch'
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()
        # adding: batch processing
        hidden_stack = hidden.repeat(max_len,1,1).contiguous().view(max_len*this_batch_size, -1) # [SxB, N]
        encoder_outputs_stack = encoder_outputs.contiguous().view(max_len*this_batch_size, -1) # [SxB, N]
        attn_energies_stack = self.score(hidden_stack.unsqueeze(2), encoder_outputs_stack.unsqueeze(1)).squeeze(1) # [SxB, 1]
        attn_energies = attn_energies_stack.view(max_len, this_batch_size).transpose(0,1)
        
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = (energy).dot(hidden)
            return energy
        
        elif self.method == 'general_batch':
            energy = self.attn(encoder_output)
            energy = (energy).bmm(hidden)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

# sequence_first
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.0):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
        else:
            self.attn = None

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        seq_len = encoder_outputs.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
        
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        if self.attn != None:
            # Calculate attention from current RNN state and all encoder outputs;
            # apply to encoder outputs to get weighted average
            attn_weights = self.attn(rnn_output, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

            # Attentional vector using the RNN hidden state and context vector
            # concatenated together (Luong eq. 5)
            rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
            context = context.squeeze(1)       # B x S=1 x N -> B x N
            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = F.tanh(self.concat(concat_input))

            # Finally predict next token
            output = F.log_softmax(self.out(concat_output).unsqueeze(0), dim=2)  # 1 x B x N?
        else:
            attn_weights = torch.zeros(1, batch_size, seq_len)
            if USE_CUDA: attn_weights = attn_weights.cuda()
            output = F.log_softmax(self.out(rnn_output), dim=2)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
