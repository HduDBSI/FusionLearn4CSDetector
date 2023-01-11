
from numpy import matmul
import torch
from torch import nn
import math
from torch.nn import *
from torch import Tensor
from typing import Optional, Any
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob import GlobalAttention
import torch.nn.functional as F
from selfattentionlayer import SelfAttention
from global_self_att import GlobalSelfAttentionLayer
from bi_lstm import LSTMModel


def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


class MyGCN(nn.Module):
    def __init__(self, hidden):
        super(MyGCN, self).__init__()
        self.GCN1 = GCNConv(hidden, hidden)
        self.GCN2 = GCNConv(hidden, hidden)
        self.GCN3 = GCNConv(hidden, hidden)
        self.mlp_gate1 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool1 = GlobalAttention(gate_nn=self.mlp_gate1)
        self.mlp_gate2 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool2 = GlobalAttention(gate_nn=self.mlp_gate2)
        self.mlp_gate3 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool3 = GlobalAttention(gate_nn=self.mlp_gate3)
        self.device = 'cuda'

        self.aaaaa = nn.Parameter(torch.empty(size=(13, 1))).to('cuda')
        nn.init.xavier_uniform_(self.aaaaa.data, gain=1.414)

    def reset_parameters(self):
        self.GCN1.reset_parameters()
        self.GCN2.reset_parameters()
        self.GCN3.reset_parameters()
        self.gpool1.reset_parameters()
        self.gpool2.reset_parameters()
        self.gpool3.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # edge_attr是权重数字还是向量？
        #print('edge_weight',edge_weight)
        edge_weight = torch.matmul(edge_attr,F.softmax(self.aaaaa,dim=0))
        #edge_weight = torch.matmul(edge_attr,self.aaaaa)
        #print("edge_weight",edge_weight.shape,edge_weight)
        edge_weight = edge_weight.reshape(len(edge_weight))
        #print("edge_weight",edge_weight.shape,edge_weight)

        x = self.GCN1(x, edge_index, edge_weight)
        x = F.relu(x)
        batch = torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        readout = [self.gpool1(x,batch=batch)]
        
        x = self.GCN2(x, edge_index, edge_weight)
        x = F.relu(x)
        batch = torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        readout += [self.gpool2(x,batch=batch)]

        x = self.GCN3(x, edge_index, edge_weight)
        x = F.relu(x)
        batch = torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        readout += [self.gpool3(x,batch=batch)]
        
        output = torch.cat(readout,dim=1)
        return output
class MyGCN_edge(nn.Module):
    def __init__(self, hidden):
        super(MyGCN_edge, self).__init__()
        self.GCN1 = GCNConv(hidden, hidden)
        self.GCN2 = GCNConv(hidden, hidden)
        self.GCN3 = GCNConv(hidden, hidden)
        self.mlp_gate1 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool1 = GlobalAttention(gate_nn=self.mlp_gate1)
        self.mlp_gate2 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool2 = GlobalAttention(gate_nn=self.mlp_gate2)
        self.mlp_gate3 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool3 = GlobalAttention(gate_nn=self.mlp_gate3)
        self.device = 'cuda'

        self.aaaaa = nn.Parameter(torch.empty(size=(13, 1))).to('cuda')
        nn.init.xavier_uniform_(self.aaaaa.data, gain=1.414)

    def reset_parameters(self):
        self.GCN1.reset_parameters()
        self.GCN2.reset_parameters()
        self.GCN3.reset_parameters()
        self.gpool1.reset_parameters()
        self.gpool2.reset_parameters()
        self.gpool3.reset_parameters()

    def forward(self, x, edge_index):
        # edge_attr是权重数字还是向量？
        #print('edge_weight',edge_weight)
        #edge_weight = torch.matmul(edge_attr,F.softmax(self.aaaaa,dim=0))
        #edge_weight = torch.matmul(edge_attr,self.aaaaa)
        #print("edge_weight",edge_weight.shape,edge_weight)
        #edge_weight = edge_weight.reshape(len(edge_weight))
        #print("edge_weight",edge_weight.shape,edge_weight)

        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        batch = torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        readout = [self.gpool1(x,batch=batch)]
        
        x = self.GCN2(x, edge_index)
        x = F.relu(x)
        batch = torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        readout += [self.gpool2(x,batch=batch)]

        x = self.GCN3(x, edge_index)
        x = F.relu(x)
        batch = torch.zeros(x.size(0),dtype=torch.long).to(self.device)
        readout += [self.gpool3(x,batch=batch)]
        
        output = torch.cat(readout,dim=1)
        return output
class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None) -> None:
        super(MyTransformerEncoder, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            src_mask: the additive mask for the src sequence (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - src_mask: :math:`(S, S)`.
            - src_key_padding_mask: :math:`(N, S)`.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return memory

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FullModel(nn.Module):
    def __init__(self, vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(FullModel, self).__init__()
        self.device = 'cuda'
        self.gcn = MyGCN(hidden)
        self.transformer = MyTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        
        self.embed=nn.Embedding(vocablen,hidden)
        #self.edge_embed=nn.Embedding(13,hidden)
        self.metric_embed=nn.Embedding(metrilen,hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.attention1 = GlobalSelfAttentionLayer(hidden, hidden, dropout, alpha)
        self.attention2 = GlobalSelfAttentionLayer(hidden, hidden, dropout, alpha)
        self.attention = GlobalSelfAttentionLayer(hidden, hidden, dropout, alpha)
        
        self.fc = nn.Linear(2*hidden, 2)

        self.W1 = nn.Parameter(torch.empty(size=(3*hidden, hidden)).to('cuda'))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)

        self.W2 = nn.Parameter(torch.empty(size=(11*hidden, hidden)).to('cuda'))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.Wo = nn.Parameter(torch.empty(size=(2*hidden, 2)).to('cuda'))
        nn.init.xavier_uniform_(self.Wo.data, gain=1.414)

    def forward(self, x, edge_index, edge_attr, metrics):
        x = self.embed(x)
        metrics = self.pos_encoder(self.metric_embed(metrics))
        out1 = self.gcn(x, edge_index, edge_attr).reshape(1,-1)
        out1 = F.elu(torch.mm(out1, self.W1))
        out2 = self.transformer(metrics).squeeze(1).reshape(1,-1)
        out2 = F.elu(torch.mm(out2, self.W2))
        fusion = torch.cat((out1,out2),dim=1)
        output =  self.fc(fusion)
        result = F.softmax(output,dim=-1)
        return result
#----------------------------------------------------------------------------------------------------------------
class BiLSTM_token_out(nn.Module):
    def __init__(self, vocablen, hidden, dropout, alpha):
        super(BiLSTM_token_out, self).__init__()
        self.device = 'cuda'

        self.token_embed=nn.Embedding(vocablen,hidden)

        self.bi_lstm = LSTMModel(hidden, 128, 2)

        self.mlp_gate = nn.Sequential(nn.Linear(2*hidden,1),nn.Sigmoid())
        self.gpool = GlobalAttention(gate_nn=self.mlp_gate)

        self.attention = GlobalSelfAttentionLayer(2*hidden, hidden, dropout, alpha)
        
        self.fc = Linear(hidden, 2)

    def forward(self, token_list):
        
        token_list = token_list.reshape(-1,1)
        
        token_list = self.token_embed(token_list)
        
        out = self.bi_lstm(token_list)

        batch = torch.zeros(out.size(0),dtype=torch.long).to(self.device)
        out = self.gpool(out, batch)
        out = self.attention(out)
        
        return out

class GCN_to_LSTM(nn.Module):
    def __init__(self, vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(GCN_to_LSTM, self).__init__()
        self.device = 'cuda'

        self.lstm_token = BiLSTM_token_out(vocablen, hidden, dropout, alpha)
        
        self.transformer = MyTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        
        self.metric_embed=nn.Embedding(metrilen,hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.attention1 = GlobalSelfAttentionLayer(hidden, hidden, dropout, alpha)
        self.attention2 = GlobalSelfAttentionLayer(11*hidden, hidden, dropout, alpha)
        
        self.fc = nn.Linear(2*hidden, 2)

    def forward(self, token_list, metrics):

        
        metrics = self.pos_encoder(self.metric_embed(metrics))
    
        out1 = self.lstm_token(token_list)
        #print('out1',out1.shape)
        out1 = self.attention1(out1)

        out2 = self.transformer(metrics).squeeze(1).reshape(1,-1)
        #print('out2',out2.shape)
        out2 = self.attention2(out2)

        fusion = torch.cat((out1,out2),dim=1)
        output =  self.fc(fusion)
        
        result = F.softmax(output,dim=-1)
        
        return result
#----------------------------------------------------------------------------------------------------------------
class myDNN_out(nn.Module):
    def __init__(self, hidden):
        super(myDNN_out, self).__init__()
        hidden = 11
        hidden_size = 64
        self.dense_layer1 = nn.Linear(hidden, hidden_size)
        self.dense_layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.dense_layer3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.dense_layer4 = nn.Linear(hidden_size//4, hidden_size//8)
        self.dense_layer5 = nn.Linear(hidden_size//8, hidden_size//16)
        self.fc = nn.Linear(hidden_size//16, 128)


    def forward(self, src_metrics):
        src_metrics = src_metrics.reshape(1,-1)
        #src_metrics = F.normalize(src_metrics,dim=0)
        #print("src_metrics",src_metrics.shape,src_metrics)
        h = self.dense_layer1(src_metrics)
        h = F.relu(h)
        h = self.dense_layer2(h)
        h = F.relu(h)
        h = self.dense_layer3(h)
        h = F.relu(h)
        h = self.dense_layer4(h)
        h = F.relu(h)
        h = self.dense_layer5(h)
        h = F.relu(h)
        out = self.fc(h)

        return out

class Transformer_to_dnn(nn.Module):
    def __init__(self, vocablen, hidden, dropout, alpha):
        super(Transformer_to_dnn, self).__init__()
        self.device = 'cuda'
        self.gcn = MyGCN(hidden)
        self.embed=nn.Embedding(vocablen,hidden)
        
        self.dnn = myDNN_out(hidden)
        self.attention1 = GlobalSelfAttentionLayer(3*hidden, hidden, dropout, alpha)
        self.attention2 = GlobalSelfAttentionLayer(hidden, hidden, dropout, alpha)
        
        self.fc = nn.Linear(2*hidden, 2)

    def forward(self, x, edge_index, edge_attr, src_metrics):

        x = self.embed(x)
    
        out1 = self.gcn(x, edge_index, edge_attr)
        out1 = self.attention1(out1)

        out2 = self.dnn(src_metrics)
        out2 = self.attention2(out2)

        fusion = torch.cat((out1,out2),dim=1)
        output =  self.fc(fusion)
        
        result = F.softmax(output,dim=-1)
        
        return result

#----------------------------------------------------------------------------------------------------------------
class BiLSTM_metric_out(nn.Module):
    def __init__(self, metrilen, hidden):
        super(BiLSTM_metric_out, self).__init__()
        self.device = 'cuda'

        self.metric_embed=nn.Embedding(metrilen,hidden)

        self.bi_lstm2 = LSTMModel(hidden, 128, 2)


    def forward(self, metrics):
        
        metrics = self.metric_embed(metrics)
        
        out2 = self.bi_lstm2(metrics)

        return out2

class Transformer_to_LSTM(nn.Module):
    def __init__(self, vocablen, metrilen, hidden, dropout, alpha):
        super(Transformer_to_LSTM, self).__init__()
        self.device = 'cuda'
        self.gcn = MyGCN(hidden)
        self.embed=nn.Embedding(vocablen,hidden)

        self.lstm_metric = BiLSTM_metric_out(metrilen, hidden)

        self.attention1 = GlobalSelfAttentionLayer(3*hidden, hidden, dropout, alpha)
        self.attention2 = GlobalSelfAttentionLayer(22*hidden, hidden, dropout, alpha)
        
        self.fc = nn.Linear(2*hidden, 2)

    def forward(self, x, edge_index, edge_attr, metrics):

        x = self.embed(x)
    
        out1 = self.gcn(x, edge_index, edge_attr)
        #print('out1',out1.shape)
        out1 = self.attention1(out1)

        out2 = self.lstm_metric(metrics).reshape(1,-1)
        #print('out2',out2.shape)
        out2 = self.attention2(out2)

        fusion = torch.cat((out1,out2),dim=1)
        output =  self.fc(fusion)
        
        result = F.softmax(output,dim=-1)
        
        return result
#----------------------------------------------------------------------------------------------------------------
class M__GCN(nn.Module):
    def __init__(self, vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(M__GCN, self).__init__()
        self.device = 'cuda'
        
        self.transformer = MyTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.metric_embed=nn.Embedding(metrilen,hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.fc = nn.Linear(hidden, 2)

        self.W2 = nn.Parameter(torch.empty(size=(11*hidden, hidden)).to('cuda'))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

    def forward(self, x, edge_index, edge_attr, metrics):
        
        metrics = self.pos_encoder(self.metric_embed(metrics))
        out2 = self.transformer(metrics).squeeze(1).reshape(1,-1)
        out2 = F.elu(torch.mm(out2, self.W2))
        output =  self.fc(out2)
        result = F.softmax(output,dim=-1)
        return result
#----------------------------------------------------------------------------------------------------------------
class M__Transformer(nn.Module):
    def __init__(self, vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(M__Transformer, self).__init__()
        self.device = 'cuda'
        self.gcn = MyGCN(hidden)
        self.embed=nn.Embedding(vocablen,hidden)
        
        self.fc = nn.Linear(hidden, 2)

        self.W1 = nn.Parameter(torch.empty(size=(3*hidden, hidden)).to('cuda'))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)

    def forward(self, x, edge_index, edge_attr, metrics):
        x = self.embed(x)
        out1 = self.gcn(x, edge_index, edge_attr).reshape(1,-1)
        out1 = F.elu(torch.mm(out1, self.W1))
        output =  self.fc(out1)
        result = F.softmax(output,dim=-1)
        return result
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

class MyModel(nn.Module):
    def __init__(self, vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(MyModel, self).__init__()
        self.device = 'cuda'
            
        #self.__gcn = M__GCN(vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha)
        #self.__transformer = M__Transformer(vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha)
        #self.FullModel = FullModel(vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha)
        #self.gcn_to_lstm = GCN_to_LSTM(vocablen, metrilen, hidden, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, alpha)
        #self.transformer_to_dnn = Transformer_to_dnn(vocablen, hidden, dropout, alpha)
        self.transformer_to_lstm = Transformer_to_LSTM(vocablen, metrilen, hidden, dropout, alpha)

    def forward(self, x, edge_index, edge_attr, metrics, token_list, src_metrics):
        
        #output = self.__gcn(x, edge_index, edge_attr, metrics)
        #output = self.__transformer(x, edge_index, edge_attr, metrics)
        #output = self.FullModel(x, edge_index, edge_attr, metrics)
        #output = self.gcn_to_lstm(token_list, metrics)
        #output = self.transformer_to_dnn(x, edge_index, edge_attr, src_metrics)
        output = self.transformer_to_lstm(x, edge_index, edge_attr, metrics)
        
        return output