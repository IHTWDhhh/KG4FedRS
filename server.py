import random

import torch
import os
import numpy as np
import torch.nn as nn
import dgl
from random import sample
from multiprocessing import Pool, Manager

from tqdm import tqdm
# from torch.multiprocessing import Pool, Manager
from model import model
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')

class server():
    def __init__(self, user_list, user_batch, users, items, embed_size, lr,neighbor_num,n_entity,n_relation,triples,weight_decay,kghop,GNNhop):
        self.n_items = len(items)
        self.user_list_with_coldstart = user_list #
        self.user_list = self.generate_user_list(self.user_list_with_coldstart) #用户交互的item数量大于1的user
        self.batch_size = user_batch
        initializer = nn.init.xavier_uniform_
        gain = nn.init.calculate_gain('relu')
        self.user_embedding = initializer(torch.empty(len(users),embed_size),gain=gain).share_memory_()
        self.item_embedding = initializer(torch.empty(len(items),embed_size),gain=gain).share_memory_()
        self.entity_embedding = initializer(torch.empty(n_entity, embed_size), gain=gain).share_memory_()
        self.model = model(embed_size, 1,kghop,GNNhop).to('cuda:0')
        self.lr = lr
        self.distribute(self.user_list_with_coldstart) #初始化本地模型：即服务端分发给所有客户端全局模型
        self.neighbor_num = neighbor_num
        self.ent_num= n_entity
        self.rel_num = n_relation
        self.embed_dim = embed_size
        self.weight_decay = weight_decay

    def generate_user_list(self, user_list_with_coldstart):
        ls = []
        for user in user_list_with_coldstart:
            if len(user.likeitems) > 0:
                ls.append(user)
        return ls

    def aggregator(self, parameter_list):
        flag = False
        number = 0
        gradient_item = torch.zeros_like(self.item_embedding)
        gradient_user = torch.zeros_like(self.user_embedding)
        loss = 0
        item_count = torch.zeros(self.item_embedding.shape[0])
        user_count = torch.zeros(self.user_embedding.shape[0])

        for parameter in parameter_list:
            [model_grad, item_grad, user_grad, returned_items, returned_users, loss_user] = parameter
            num = len(returned_items)
            item_count[returned_items] += num
            user_count[returned_users] += num
            loss += loss_user ** 2 * num

            number += num
            if not flag:
                flag = True
                gradient_model = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad
                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i] * num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad
                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i] * num
        loss = torch.sqrt(loss / number)
        print('trianing average loss:', float(loss))
        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)

        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number
        return gradient_model, gradient_item, gradient_user

    def distribute(self, users):
        for user in users:
            user.update_local_GNN(self.model)

    def distribute_one(self, user):
        user.update_local_GNN(self.model)

    def predict(self, valid_data):
        # print('predict')
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []
        self.distribute([self.user_list_with_coldstart[i] for i in set(users)])

        for i in range(len(users)):
            res_temp = self.user_list_with_coldstart[users[i]].predict(items[i], self.user_embedding, self.item_embedding)
            res.append(float(res_temp))
        return np.array(res)

    def train_one(self, user, user_embedding, item_embedding):
        return user.train(user_embedding, item_embedding)

    def distribute_neighbor(self,taruser,users):
        dict={}
        usersindex = 0
        for user in users:
            if user != taruser:
                dict[user.id_self] = [self.Isneighbor(user,taruser),usersindex]
            usersindex += 1
        dictsort = sorted(dict.items(), key=lambda d: len(d[1][0]), reverse=True)[:self.neighbor_num]
        anslist = [i[0] for i in dictsort if len(i[1][0])!=0]
        neighboritems = [dict[i][0] for i in anslist]
        taruser.change_nighbor(anslist,neighboritems)
    def reduce_LR(self,index):
        self.lr =self.lr*index
        print("server reduce the lr,now the lr = ", self.lr)
    def Isneighbor(self,user,taruser):
        reslist = []
        num = 0
        for i in user.items:
            if i in taruser.items :
                reslist.append(i)
        return reslist

    def distribute_KGE(self,user):
        user.get_entity_embedding(self.entity_embedding[user.entity_id])

    def choose_user_client(self,user_list,batch_size):
        for i in range(0, len(user_list), batch_size):
            yield user_list[i:i + batch_size]
    def gen_uid_item(self,users):
        uid_item=[(user.id_self,user.likeitems) for user in users]
        return uid_item
    def train(self):
        """服务端 将全局模型分发给选择训练的本地模型 只有被选择训练了的客户端才能够借助全局模型更新本地模型"""
        shuffled_list = self.user_list.copy()  # 复制列表以避免修改原始列表
        random.shuffle(shuffled_list)
        for users in self.choose_user_client(shuffled_list,self.batch_size):
            parameter_list = []
            self.distribute(users)
            """开始本地训练"""
            for user in users:
                parameter_list.append(user.train(self.user_embedding, self.item_embedding))
            gradient_model, gradient_item, gradient_user= self.aggregator(parameter_list)
            ls_model_param = list(self.model.parameters())
            item_index = gradient_item.sum(dim = -1) != 0
            user_index = gradient_user.sum(dim = -1) != 0
            i = 0
            for param in self.model.parameters():
                # 这里修改模型的参数时，server上的模型参数就被修改了
                param.data = param.data - (self.lr * gradient_model[i]).to('cuda:0')
                i += 1
            self.item_embedding[item_index] = self.item_embedding[item_index] -  self.lr * gradient_item[item_index]
            self.user_embedding[user_index] = self.user_embedding[user_index] -  self.lr * gradient_user[user_index]
