import time

import numpy as np
import pandas as pd
import dgl
import torch
from random import sample
import random
class thirdserver():
    def __init__(self,n_entity, n_relation, triples,item,neighbor_num,dataname):
        self.triples = triples
        self.kgdf = pd.DataFrame(triples, columns=['h', 'r', 't'])
        self.itemdata = item
        self.n_relation = n_relation
        self.neighbor_num = neighbor_num
        self.cache={}
        self.multiple = [1]
        self.dataname = dataname
    def judege_item(self,id):
        if id in self.itemdata:
            return 'item'
        else:
            return 'entity'
    def Get_Neighbor(self, selfuid, uitems, uids_items):
        uneighbor = [(uid,set(items) & set(uitems)) for uid,items in uids_items if uid != selfuid]
        dictsort = sorted(uneighbor, key=lambda d: len(d[1]), reverse=True)[:self.neighbor_num]
        anslist = []
        neighboritems = []
        for i in range(len(dictsort)):
            nitems =list(dictsort[i][1])
            if len(nitems) == 0 and i == 0:
                return anslist, neighboritems
            else:
                anslist.append(dictsort[i][0])
                neighboritems.append(nitems)
        return anslist, neighboritems

    def get_one_hop_kgneighbor(self, items, dir='in', hop=0):
        if hop % 2 == 0:
            triples = self.kgdf[self.kgdf['h'].isin(items)]
            nextid = triples['t'].tolist()

        else:
            triples = self.kgdf[self.kgdf['t'].isin(items)]
            nextid = triples['h'].tolist()
        result = nextid
        return result

    def Get_1hop_KGE(self, uid, uitems,hop):
        tail_dict = {}
        tails = set()
        tails_items = set()

        nextid = self.get_one_hop_kgneighbor(items=uitems, hop=hop)
        # tail_dict[i] = nextid
        tails.update(nextid)
        tails_items = tails.intersection(self.itemdata)
        return tail_dict, list(tails_items), list(tails)

    def matchclient(self,uid,u_items,likeitems,uids_items,hop):
        itemsnum = len(u_items)
        alltail_dict, alltails_items, alltails = [],[],[]
        alltails_items.append(likeitems)
        allnighborlist,allneighboritems = [],[]
        # 在CF图里面使用uitems来进行交互
        nighborlist, neighboritems = self.Get_Neighbor(uid, u_items, uids_items)
        allnighborlist.append(nighborlist)
        allneighboritems.append(neighboritems)
        # 在知识图谱里面链接的时候使用user交互的物品
        uitems = likeitems
        onlyone = set(likeitems)
        for i in range(hop*2):
            _, tails_items, tails = self.Get_1hop_KGE(uid,uitems,i)

            tails_items = [int(items) for items in tails_items if items not in onlyone]
            # alltail_dict.append(tail_dict)
            # 电影那个数据集非常的稠密需要限制kgitems的数量
            multiple = random.choice(self.multiple)
            if int(itemsnum * multiple) < len(tails_items):
                tails_items = sample(tails_items, int(itemsnum * multiple))
            alltails_items.append(tails_items)
            onlyone.update(tails_items)
            alltails.append(tails)
            if len(alltails_items) % 2 == 1:
                nighborlist, neighboritems = self.Get_Neighbor(uid, tails_items, uids_items)
                allnighborlist.append(nighborlist)
                allneighboritems.append(neighboritems)
            uitems = tails
        alltails_items = [alltails_items[i] for i in range(len(alltails_items)) if i !=0]
        return alltail_dict,alltails_items,allnighborlist,allneighboritems