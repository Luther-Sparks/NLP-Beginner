import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sort_by_seq_len, masked_softmax, weighted_sum
import transformers

class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs
    """
    def forward(self, sequences_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch (torch.Tensor(batch, sequences_length, embedding_dim)): A batch of sequences of input to an RNN layer.

        Returns:
            torch.Tensor: A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = F.dropout(ones, p = self.p, training=self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch
        
class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_module : nn.RNNBase, input_size : int, hidden_size : int, 
                 num_layers=1 , bias=True, dropout=0.5, bidirectional=False):
        """ 
        RNN layer

        Args:
            rnn_module (nn.RNNBase): A type of RNN
        """
        self.encoder = rnn_module(input_size, hidden_size, num_layers, bias, batch_first=True, dropout=dropout,bidirectional=bidirectional)
    
    def forward(self, sequences_batch : torch.Tensor, sequences_lengths : torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            sequences_batch (torch.Tensor(batch, sequence, vector_dim)): A batch of variable length sequences.
            sequences_lengths (torch.Tensor): A 1D tensor containing the sizes of the sequences in the input batch.

        Returns:
            torch.Tensor: The outputs (hidden states) of the encoder for the sequences in the input batch.
        """
        # notice that to use nn.utils.rnn.pack_padded_sequence function, we have to sort batch by length in descent order.
        # Later try to use it without sort.
        sorted_batch, sorted_lengths, restoration_idx = sort_by_seq_len(sequences_batch, sequences_lengths, True)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        
        outputs, _ = self.encoder(packed_batch) # pass it to RNN layer
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        ordered_outputs = outputs.index_select(0, restoration_idx)
        return ordered_outputs

class SoftmaxAttention(nn.Module):
    """
        Attention layer. 
        The particle uses softmax attention. I learn from the Internet that some special attention may be fast. Maybe I should try them if I'm available.
    """
    def forward(self, premise_batch : torch.Tensor, premise_mask : torch.Tensor, hypothesis_batch : torch.Tensor, hypothesis_mask : torch.Tensor):
        similar_matrix = premise_batch.matmul(hypothesis_batch.transpose(2, 1).contiguous())
        
        pre_hyp_attention = masked_softmax(similar_matrix, hypothesis_mask)
        hyp_pre_attention = masked_softmax(similar_matrix.transpose(1, 2).contiguous(), premise_mask)
        
        attended_premise = weighted_sum(hypothesis_batch, pre_hyp_attention, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_pre_attention, hypothesis_mask)
        
        return attended_premise, attended_hypotheses
        
class Pooling(nn.Module):
    """
        Pooling Layer
    """
    def __init__(self):
        self.avg_pooling = nn.AvgPool1d(1, 1)
        self.max_pooling = nn.MaxPool1d(1, 1)
    
    def forward(self, premise, hypotheses):
        premise_avg = self.avg_pooling(premise)
        premise_max = self.max_pooling(premise)
        hypotheses_avg = self.avg_pooling(hypotheses)
        hypotheses_max = self.max_pooling(hypotheses)
        return torch.cat(premise_avg, premise_max, hypotheses_avg, hypotheses_max)