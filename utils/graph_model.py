import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_scatter
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.nn.pool import TopKPooling, EdgePooling
import numpy as np

# 指标评估
def macro_f1(label,prediction)  -> float:

    weights =  [3  /  7,  2  /  7,  1  /  7,  1  /  7]
    macro_F1 =  0.
    for i in  range(len(weights)):
        TP =  np.sum((label==i) & (prediction==i))
        FP =  np.sum((label!= i) & (prediction == i))
        FN =  np.sum((label == i) & (prediction!= i))
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FN)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
        
        print('Task %d:\n Prcesion %.2f, Recall %.2f, F1 %.2f' % (i+1, precision, recall, F1))
        
    return macro_F1

class Log_Rep_Graph(pl.LightningModule):
    
    def __init__(self, input_dim, hidden_dim, learning_rate=1e-3):
        super(Log_Rep_Graph, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.pool1 = TopKPooling(in_channels=hidden_dim, ratio=0.5)
        self.conv2 = GCNConv(hidden_dim, 4)
        
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
        self.lr = learning_rate
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5)
        x = F.relu(x)
        pool_out = self.pool1(x, data.edge_index, batch=data.batch)
        x, edge_index, edge_attr, batch_pool, perm, score = pool_out
        x = self.conv2(x, edge_index)

        # pooling
        x = torch_scatter.scatter_mean(src=x, index=batch_pool, dim=0)
        x = F.softmax(x,dim=1)
        
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=5e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(self.logsoftmax_func(out), batch.label)
        self.log('loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(self.logsoftmax_func(out), batch.label)
        self.log('val_loss', loss)
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(self.logsoftmax_func(out), batch.label)
        self.predictions['pred'].extend(out.cpu().numpy())
        self.predictions['id'].extend(batch.id)
        self.predictions['label'].extend(batch.label.cpu().numpy())
        return {'test_loss': loss}

class Log_Rep_GraphAttention(Log_Rep_Graph):
    
    def __init__(self, input_dim, hidden_dim, num_head, learning_rate, *args, **kwargs):
        super(Log_Rep_GraphAttention, self).__init__(input_dim, hidden_dim, learning_rate, *args, **kwargs)
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_head)
        self.pool1 = TopKPooling(in_channels=hidden_dim * num_head, ratio=0.5)
        self.conv2 = GCNConv(hidden_dim * num_head, 4)
        
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        pool_out = self.pool1(x, data.edge_index, batch=data.batch)
        x, edge_index, edge_attr, batch_pool, perm, score = pool_out
        x = self.conv2(x, edge_index)
        
        # pooling
        x = torch_scatter.scatter_mean(src=x, index=batch_pool, dim=0)
        x = F.softmax(x,dim=1)
        
        return x

class Log_Rep_GraphConv(Log_Rep_Graph):
    
    def __init__(self, input_dim, hidden_dim,learning_rate, *args, **kwargs):
        super(Log_Rep_GraphConv, self).__init__(input_dim, hidden_dim, learning_rate, *args, **kwargs)
        
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.pool1 = TopKPooling(in_channels=hidden_dim, ratio=0.5)
        self.conv2 = GCNConv(hidden_dim, 4)
        
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        pool_out = self.pool1(x, data.edge_index, batch=data.batch)
        x, edge_index, edge_attr, batch_pool, perm, score = pool_out
        x = self.conv2(x, edge_index)
        
        # average pooling?
        x = torch_scatter.scatter_mean(src=x, index=batch_pool, dim=0)
        x = F.softmax(x,dim=1)
        
        return x

