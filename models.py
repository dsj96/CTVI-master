import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from attention import Multi_Head_SelfAttention
import math


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, degree):
        super(SGC, self).__init__()
        self.degree = degree
        self.W = nn.Linear(nfeat, nclass, bias=True)
        self.alpha = nn.Parameter(Variable(torch.FloatTensor([1 for i in range(self.degree)]), requires_grad=True))

    def init(self):
        stdv = 1. / math.sqrt(self.alpha.weight.size(0))
        self.alpha.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, weight_adj_list):
        if type(weight_adj_list) == list:
            weight_aug_adj = 0
            for idx,item in enumerate(weight_adj_list):
                weight_aug_adj = weight_aug_adj + self.alpha[idx] * item
            return self.W(torch.spmm(weight_aug_adj, x))
        else:
            return self.W(weight_adj_list)

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input) # XW
        output = torch.spmm(adj, support) # AXW
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout, degree):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.degree = degree

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class HANGZHOU_model(nn.Module):
    """
    multi_slice model.
    """
    def __init__(self, model_type, num_head, num_slice, nfeat, nhid, nclass, dropout, degree):
        super(HANGZHOU_model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.num_head = num_head
        self.num_slice = num_slice
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.degree = degree

        self.model_list = []


        for i in range(self.num_slice):
            self.model_list.append(
                get_model(self.model_type, self.nfeat, self.nclass, self.degree, self.nhid, self.dropout)
            )
        self.attention = Multi_Head_SelfAttention(num_head=self.num_head, num_vocab=self.num_head, input_dim=self.nclass, hidden_dim=self.nclass, out_dim=self.nclass)


    def forward(self, x, adj, use_relu=True):
        output_list = []
        for i in range(self.num_slice):
            cur_output = self.model_list[i](x, adj).unsqueeze(0)
            output_list.append(cur_output)

        output = torch.cat(output_list,dim=0)   # (12,553,128)
        output = output.transpose(0,1)          # (553,12,128)
        attention_output_list = []
        for i in range(self.num_slice-self.num_head):
            if i==0:
                attention_output = self.attention(output[:,i:i+self.num_head,:]).transpose(0,1)[0:self.num_head-1] # (553,3,128)->(3,553,128)->unsqueeze(0)
                attention_output_list.append(attention_output)
            if i==self.num_slice-self.num_head-1:
                attention_output = self.attention(output[:,i:i+self.num_head,:]).transpose(0,1)[-2:]  # (553,3,128)->(3,553,128)->unsqueeze(0)
                attention_output_list.append(attention_output)
            else:
                attention_output = self.attention(output[:,i:i+self.num_head,:]).transpose(0,1)[1:2]  # (553,3,128)->(3,553,128)->unsqueeze(0)
                attention_output_list.append(attention_output)
        return torch.cat(attention_output_list,dim=0)


class JINAN_model(nn.Module):
    """
    multi_slice model.
    """
    def __init__(self, model_type, num_head, num_slice, nfeat, nhid, nclass, dropout, degree):
        super(JINAN_model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.num_head = num_head
        self.num_slice = num_slice
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.degree = degree
        self.model_list = []

        for i in range(self.num_slice):
            self.model_list.append(
                get_model(self.model_type, self.nfeat, self.nclass, self.degree, self.nhid, self.dropout)
            )
        self.attention = Multi_Head_SelfAttention(num_head=self.num_head, num_vocab=self.num_head, input_dim=self.nclass, hidden_dim=self.nclass, out_dim=self.nclass)

    def return_attention_index(self, idx):

        if idx== 0 or idx==1 or idx==2:
            return [0,1,2]
        elif idx < 12*1:
            return [idx-2, idx-1, idx]
        elif idx < 12*2: # 24*2
            return [idx-12, idx-2, idx-1, idx]
        elif idx < 12*7: # 168
            return [idx-12*2, idx-12, idx-2, idx-1, idx]
        elif idx < 12*14: # 336
            return [idx-12*7, idx-12*2, idx-12, idx-2, idx-1, idx]
        else:
            return [idx-12*14, idx-12*7, idx-12*2, idx-12, idx-2, idx-1, idx] # TODO: 12 slice


    def forward(self, x, adj, use_relu=True):

        output_list = []
        for i in range(self.num_slice):
            cur_output = self.model_list[i](x, adj).unsqueeze(0)
            output_list.append(cur_output)

        output = torch.cat(output_list,dim=0)   # (774,433,128)
        output = output.transpose(0,1)          # (433,774,128)
        attention_output_list = []
        for idx in range(self.num_slice):
            cur_idx = self.return_attention_index(idx)
            if idx==0 or idx==1 or idx==2:
                attention_output = self.attention(output[:,cur_idx,:]).transpose(0,1)[idx] # (433,3,128)->(3,433,128)
                attention_output_list.append(attention_output.unsqueeze(0))
            else:
                attention_output = self.attention(output[:,cur_idx,:]).transpose(0,1)[-1]  # (433,x,128)->(x,433,128)->unsqueeze(0)
                attention_output_list.append(attention_output.unsqueeze(0))
        return torch.cat(attention_output_list,dim=0)


def get_model(model_opt, nfeat, nclass, degree, nhid=128, dropout=0, cuda=True): # ('SGC', 1433, 7, 0, 0, False)
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout,
                    degree=degree)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass,
                    degree=degree) # (1433, 7)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    # if cuda: model.cuda()
    return model
