import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv
import dgl
import dgl.function as fn
class model(nn.Module):
    def __init__(self, embed_size, head_num,kghop,GNNhop):
        super().__init__()
        self.GNNhop = GNNhop
        self.GAT_layer = self.init_GATmodel( embed_size, head_num,kghop=kghop,GNNhop = GNNhop)
        self.GAT_layer1 = GATConv(embed_size, embed_size, head_num)
        self.gcn = GCNLayer()
        self.linear=nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
        )
    def init_GATmodel(self, embed_size, head_num, kghop, GNNhop):
        gat_layers = nn.ModuleList()
        for i in range(kghop * GNNhop):
            temp = GATConv(embed_size, embed_size, head_num)
            gat_layers.append(temp)
        return gat_layers

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def get_userembedding(self,features):
        n = features.shape[1]
        user_embedding = features[0, :].unsqueeze(1).reshape(1, n)
        return user_embedding
    def forward(self, UIgraph,UEgraph,features1,UEfeatures,kghop):
        user_embeddings = self.get_userembedding(features1)
        CFfeatures = []
        for i in range(self.GNNhop):
            features1 = self.gcn(UIgraph, features1)
            CFfeatures+=[features1]
        CFfeatures = torch.stack(CFfeatures, dim=0)
        featuressum = torch.sum(CFfeatures,dim=0)

        user_embedding1 = self.get_userembedding(featuressum)
        user_embeddings = torch.cat([user_embeddings,user_embedding1])
        i=0
        FakeCFfeatures = []
        for graph ,feature in zip(UEgraph,UEfeatures):
            graph = graph.to('cuda:0')
            if graph.num_nodes() != 0:
                for j in range(self.GNNhop):
                    gat = self.GAT_layer[i+j]
                    features_1= gat(graph, feature)
                    n = features_1.shape[0]
                    features_1 = features_1.reshape(n, -1)
                    FakeCFfeatures+=[features_1]

                temp = torch.stack(FakeCFfeatures[i:i+self.GNNhop], dim=0)
                user_embedding2 = self.get_userembedding(torch.sum(temp, dim=0))
                user_embeddings = torch.cat([user_embeddings,user_embedding2])
            else:
                for j in range(self.GNNhop):
                    FakeCFfeatures.append([1])
            i+=self.GNNhop
        user_embeddings = self.linear(user_embeddings)
        user_embedding = torch.sum(user_embeddings,dim=0)
        return user_embedding,featuressum

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, features):
        with graph.local_scope():
            node_f = features
            # D^-1/2
            degs = graph.out_degrees().to("cuda:0").float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(message_func=fn.copy_u('n_f', 'm'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to("cuda:0").float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm
            return rst
