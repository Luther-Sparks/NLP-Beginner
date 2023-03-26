from datasets import Dataset, load_dataset
from layers import Seq2SeqEncoder, SoftmaxAttention, Pooling
import torch
from torch import nn
import torch.nn.functional as F
from util import get_mask

snli = load_dataset('snli')

class ESIM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int = 64,
        vocab_size: int = 1973,
        label_num: int = 3,
        dropout: float = 0.5,
        device: str = 'gpu',
    ):
        
        self.device = device
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.embedding_layer = nn.Embedding(embedding_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_layer = Seq2SeqEncoder(nn.LSTM, embedding_size, hidden_size, bidirectional=True)
        self.inference_layer = SoftmaxAttention()
        self.composition_layer = nn.Sequential(
            Seq2SeqEncoder(nn.LSTM, 4*2*self.hidden_size, 4*2*self.hidden_size, bidirectional=True),
            nn.Linear(4*2*hidden_size, self.hidden_size),
            nn.ReLU()
        )
        self.pooling_layer = Pooling()
        self.classification_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(2*4*self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.label_num)
        )
        self.apply(_init_esim_weights)
        
        def forward(self, premises, premises_lengths, hypotheses, hypotheses_lengths):
            premises_mask = get_mask(premises, premises_lengths)
            hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)
            embedded_premises = self.embedding_layer(premises)
            embedded_hypotheses = self.embedding_layer(hypotheses)
            embedded_premises = self.dropout(embedded_premises)
            embedded_hypotheses = self.dropout(embedded_hypotheses)
            encoded_premises = self.encoding_layer(embedded_premises, premises_lengths)
            encoded_hypotheses = self.encoding_layer(embedded_hypotheses, hypotheses_lengths)
            attended_premises, attended_hypotheses = self.inference_layer(encoded_premises, premises_mask, encoded_hypotheses, hypotheses_mask)
            enhanced_premises = torch.cat([encoded_premises, attended_premises, encoded_premises-attended_premises, encoded_premises*attended_premises], dim=-1)
            enhanced_hypotheses = torch.cat([encoded_hypotheses, attended_hypotheses, encoded_hypotheses-attended_hypotheses, encoded_hypotheses*attended_hypotheses], dim=-1)
            vector_premises = self.composition_layer(enhanced_premises)
            vector_hypotheses = self.composition_layer(enhanced_hypotheses)
            result_vector = self.pooling_layer(vector_premises, vector_hypotheses)
            logits = self.classification_layer(result_vector)
            probabilities = F.softmax(logits, dim=-1)
            
            return logits, probabilities
            

def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0