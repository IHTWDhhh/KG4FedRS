# import matplotlib.pyplot as plt
import math

import torch
import copy
from random import sample
import torch.nn as nn
import numpy as np
import dgl
import pdb
from model import model
class user():
    def __init__(self, id_self, items, ratings, neighbors, embed_size, clip, lr, l2_weight, laplace_lambda, negative_sample,thirdserver,kghop,GNNhop,sampleradio):
        self.negative_sample = negative_sample
        self.clip = clip
        self.lr = lr
        self.l2_weight = l2_weight
        self.laplace_lambda = laplace_lambda
        self.id_self = id_self
        self.items = items
        self.embed_size = embed_size
        self.ratings = ratings
        likeitems,dlitems = self.Genlikeitems(items,ratings)
        self.likeitems=likeitems[:int(len(likeitems)*sampleradio)]
        dlitems = dlitems[:int(len(dlitems)*sampleradio)]
        self.items=self.likeitems+dlitems
        self.ratings=[1.0 for i in self.likeitems]+[0.0 for i in dlitems]
        self.neighbors = neighbors
        self.model = model(embed_size, 1,kghop,GNNhop).cuda()
        self.user_feature = None
        self.thirdserver = thirdserver
        self.kghop = kghop
        self.flag=False

    def Genlikeitems(self,items,ratings):
        likeitems = []
        dlitems=[]
        for i in zip(items,ratings):
            if i[1]>0:
                likeitems.append(i[0])
            else:
                dlitems.append(i[0])
        return likeitems,dlitems

    def build_local_UIgraph(self, id_self, items, neighbors,itemlist):
        G = dgl.DGLGraph()
        neighborlist = []
        dic_item = {}
        # neighbor->item->self
        # feature顺序 self-neighbor-item
        for item in range(len(items)):
            dic_item[items[item]] = 1 + len(neighbors) + item
        # 本地添加item->self
        G.add_edges(list(dic_item.values()), 0)
        newneighboritem = []
        for i in range(len(itemlist)):
            for j in range(len(itemlist[i])):
                newneighboritem.append(dic_item[itemlist[i][j]])
        for j in range(len(neighbors)):
            neighborlist.extend([j + 1 for temp in range(len(itemlist[j]))])
        G.add_edges(neighborlist, newneighboritem)
        G = dgl.to_bidirected(G)
        G = dgl.add_self_loop(G)
        return G

    def build_local_UEgraph(self, id_self, items, neighbors,itemlist):
        G = dgl.DGLGraph()
        neighborlist = []
        dic_item = {}
        # neighbor->item->self
        # feature顺序 self-neighbor-item
        for item in range(len(items)):
            dic_item[items[item]] = 1 + len(neighbors) + item
        # 本地添加item->self
        G.add_edges(list(dic_item.values()), 0)
        newneighboritem = []
        for i in range(len(itemlist)):
            for j in range(len(itemlist[i])):
                newneighboritem.append(dic_item[itemlist[i][j]])
        for j in range(len(neighbors)):
            neighborlist.extend([j + 1 for temp in range(len(itemlist[j]))])
        G.add_edges(neighborlist, newneighboritem)
        G = dgl.to_bidirected(G)
        G = dgl.add_self_loop(G)
        # self.drawgraph(G)
        return G

    def user_embedding(self, embedding):
        neightbortensor = []
        userself = embedding[torch.tensor(self.id_self)]
        for i in self.neighbors:
            if len(i) == 0:
                neightbortensor.append(torch.tensor([]))
            else:
                neightbortensor.append(embedding[torch.LongTensor(i)])
        return userself,neightbortensor

    def item_embedding(self, embedding):
        entitytensor = []
        likeembedding= torch.tensor([])
        itemembedding = embedding[torch.tensor(self.items)]
        if len(self.likeitems)!=0:
            likeembedding = embedding[torch.tensor(self.likeitems)]
        for i in range(len(self.entity_id)):
            if i % 2 == 0:
                continue
            if len(self.entity_id[i]) == 0:
                entitytensor.append(torch.tensor([]))
            else:
                entitytensor.append(embedding[torch.LongTensor(self.entity_id[i])])
        return itemembedding,entitytensor,likeembedding

    def send_items(self, uids_items):
        if self.flag!=False:
            return
        # 从第三方服务器上面获取知识图谱数据，和实体id
        _, self.entity_id, self.neighbors, nighbor_itemlist = self.thirdserver.matchclient(self.id_self,
                                                                                           self.likeitems,
                                                                                           self.likeitems,
                                                                                           uids_items,
                                                                                           self.kghop)
        items = copy.deepcopy(self.likeitems)
        self.UIgraph = self.build_local_UIgraph(self.id_self, items, self.neighbors[0], nighbor_itemlist[0])
        self.UEgraph = []
        index = 1
        for i in range(1, self.kghop + 1):
            self.UEgraph.append(self.build_local_UIgraph(self.id_self, self.entity_id[index], self.neighbors[i],
                                                         nighbor_itemlist[i]))
            index += 2
        self.flag=True
    def GenItemEmbeddings(self,items_embedding,like_items_embedding):
        index = 0
        for i in range(len(self.items)):
            if self.ratings[i] > 0 :
                items_embedding[i] = like_items_embedding[index]
                index += 1
            else:
                continue
        return items_embedding
    def GNN(self, embedding_user, embedding_item):
        # 取出邻居向量和自身的向量
        self_embedding,neighbor_embeddinglist  = self.user_embedding(embedding_user)
        # 取出自身交互过的物品的向量
        items_embedding,neighbor_item_embeddinglist,likeembedding = self.item_embedding(embedding_item)
        # 现在的顺序 uitem-neighboritem-self-neighborU
        features1 =  torch.cat((self_embedding.unsqueeze(0), neighbor_embeddinglist[0],likeembedding), 0).cuda()
        UEfeaturs=[]
        for i in range(self.kghop):
            UEfeaturs.append(torch.cat((self_embedding.unsqueeze(0),neighbor_embeddinglist[i+1],neighbor_item_embeddinglist[i]),0).cuda())

        # 进入 model.forward里面
        user_feature,featuresum = self.model(self.UIgraph.to('cuda:0'),self.UEgraph,features1,UEfeaturs,self.kghop)

        like_items_embedding = featuresum[1+neighbor_embeddinglist[0].shape[0]:]
        user_feature = user_feature.cpu()

        self.user_feature = user_feature.detach()
        # 只返回了用户的向量，没有返回物品的向量。

        return user_feature, embedding_item


    #更新本地模型到全局上
    def update_local_GNN(self, global_model):
        self.model = copy.deepcopy(global_model)

   #用户端的损失函数计算
    def Loss(self,users, items):
        train_items = items[torch.tensor(self.items)]
        batch_size = train_items.shape[0]
        users = users.repeat(batch_size,1)
        predicted = (train_items * users).sum(dim=1)
        predicted = torch.sigmoid(predicted)
        ratings = torch.FloatTensor(self.ratings)
        loss = nn.BCELoss()

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.l2_weight * regularizer / batch_size
        return loss(predicted, ratings)+emb_loss


    def predict(self, item_id, embedding_user, embedding_item):
        if self.user_feature !=None:
            userembedding = self.user_feature
        else:
            userembedding = embedding_user[self.id_self]
        item_embedding = embedding_item[item_id]
        return torch.matmul(userembedding, item_embedding.t())

    def topk_predict_U(self, embedding_user, embedding_item):
        if self.user_feature == None:
            self.user_feature = embedding_user[self.id_self]
        item_embedding = embedding_item
        return torch.matmul(self.user_feature,item_embedding.t())

    #负采样
    def negative_sample_item(self, grad):
        item_num, embed = grad.shape
        ls = [i for i in range(item_num) if i not in self.items]
        # sampled_items = sample(ls, self.negative_sample)
        grad_value = torch.masked_select(grad, grad != 0)
        mean = torch.mean(grad_value)
        var = torch.std(grad_value)

        returned_items = copy.deepcopy(self.items)
        entity_id  = [i for entity in self.entity_id if len(entity) != 0 for i in entity]
        returned_items.extend(entity_id)
        return returned_items

    #本地差分隐私的干扰
    def LDP(self, tensor):
        # tensor = tensor.cpu()
        # tensor_mean = torch.abs(torch.mean(tensor))
        # tensor = torch.clamp(tensor, min = -self.clip, max = self.clip)
        # noise = np.random.laplace(0, tensor_mean * self.laplace_lambda,size=tensor.shape).astype(np.float32)
        # tensor += noise
        return tensor

    #本地模型的训练
    def train(self, embedding_user, embedding_item):
        embedding_user = torch.clone(embedding_user).detach()
        embedding_item = torch.clone(embedding_item).detach()
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        self.model.train()
        user_feature, items_embedding= self.GNN(embedding_user, embedding_item)
        loss = self.Loss(user_feature, items_embedding)
        self.model.zero_grad()
        loss.backward()
        model_grad = []
        for param in list(self.model.parameters()):
            grad = copy.deepcopy(param.grad)
            if grad == None:
                grad = torch.zeros_like(param)
            grad = self.LDP(grad)
            model_grad.append(grad)
        # 物品伪交互采样
        returned_items = self.negative_sample_item(embedding_item.grad)
        item_grad = self.LDP(embedding_item.grad[returned_items, :])
        returned_users = [self.id_self]
        user_grad = embedding_user.grad[returned_users, :]
        res = (model_grad, item_grad, user_grad, returned_items, returned_users,loss.detach())
        return res
