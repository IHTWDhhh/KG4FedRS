import pickle
import torch
import numpy as np
import utils
from data_loader import load_kg
from logger import Logger
from user import user
from server import server
from thirdserver import thirdserver
from sklearn import metrics
import math
import argparse
import warnings
import sys
import faulthandler
from sklearn.metrics import roc_auc_score,f1_score
from tqdm import tqdm
import multiprocessing
faulthandler.enable()
warnings.filterwarnings('ignore')
# torch.multiprocessing.set_sharing_strategy('file_system')

CORES = multiprocessing.cpu_count() // 2
def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res, dtype=int)

def ctr_eval(server, valid_data):
    label = valid_data[:, -1]

    predicted = server.predict(valid_data)
    predicted = torch.tensor(predicted)
    predicted = torch.sigmoid(predicted)

    auc = roc_auc_score(y_true=label, y_score=predicted)

    prediction = [1 if i >= 0.5 else 0 for i in predicted]

    f1 = f1_score(y_true=label, y_pred=prediction)
    acc = np.mean(np.equal(prediction, label))
    # mae = sum(abs(label - predicted)) / len(label)
    # rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))
    return auc, f1, acc
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in [20]:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

def getUserPosItems(train_data):
    res = []
    for key in train_data.keys():
        if len(train_data[key]) > 0:
            items = []
            for ratings in train_data[key]:
                item, rate, _ = ratings
                if rate > 0:
                    items.append(int(item))
            if len(items) > 0:
                res.append(items)
    return res
if __name__ == '__main__':

    faulthandler.enable()
    warnings.filterwarnings('ignore')
    # torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description="args for KG4FedRs")
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
    parser.add_argument('--data', default='movie-1m')
    parser.add_argument('--user_batch', type=int, default=256)
    parser.add_argument('--clip', type=float, default=0)
    parser.add_argument('--laplace_lambda', type=float, default=0)
    parser.add_argument('--negative_sample', type=int, default=0)
    parser.add_argument('--valid_step', type=int, default=1)
    parser.add_argument('--lr_step', type=int, default=11)
    parser.add_argument('--stop_step', type=int, default=5)
    parser.add_argument('--neighbor_num', type=int, default=200)
    parser.add_argument('--kg_hop', type=int, default=1)
    parser.add_argument('--GNNhop', type=int, default=2)
    parser.add_argument('--sampleradio',type=float,default=1)
    args = parser.parse_args()
    print("{}clip_{}lamda_{}neigh-{}batch_{}epoch_{}lr_{}dim_{}kg_hop_{}GNNhop_{}sampleradio".format(args.clip, args.laplace_lambda, args.neighbor_num, args.user_batch, args.valid_step, args.lr,
        args.embed_size, args.kg_hop,args.GNNhop,args.sampleradio))
    embed_size = args.embed_size
    user_batch = args.user_batch
    lr = args.lr
    l2_weight = args.l2_weight

    import time

    t = './logs/{}/{}clip_{}lamda_{}neigh-{}batch_{}epoch_{}lr_{}dim_{}kg_hop_{}GNNhop_{}sampleradio'.format(
        args.data, args.clip, args.laplace_lambda, args.neighbor_num, args.user_batch, args.valid_step, args.lr,
        args.embed_size, args.kg_hop,args.GNNhop,args.sampleradio)
    T = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    filename = t + T + '.txt'
    log = Logger(filename)
    sys.stdout = log

    abspath = "./data/"
    # read data
    if args.testflag == 1:
        data_file = open(abspath + args.data + '/PerFedratings.pkl', 'rb')
    else:
        data_file = open(abspath + args.data + '/PerFedratings.pkl', 'rb')

    print('using dataset:{},mession: {} (0 for ctr, 1 for topk)'.format(args.data,args.testflag))
    [train_data, valid_data, test_data, user_id_list, item_id_list] = pickle.load(data_file)
    data_file.close()
    allpos = []

    valid_data = processing_valid_data(valid_data)
    test_data = processing_valid_data(test_data)
    train_data_test = processing_valid_data(train_data)

    # get kg data
    n_entity, n_relation, triples = load_kg(args)

    # build thirdserver
    thirdserver = thirdserver(n_entity, n_relation, triples, item_id_list, args.neighbor_num,args.data)

    # build user_list
    user_list = []
    for u in tqdm(user_id_list):
        neighbor = []
        ratings = train_data[u]
        items = []
        rating = []
        for i in range(len(ratings)):
            item, rate, _ = ratings[i]
            items.append(item)
            rating.append(rate)
        # 这里完成了数据的处理 对 user 进行了实例化
        user_list.append(user(u, items, rating, neighbor,
                              embed_size, args.clip, lr, l2_weight, args.laplace_lambda, args.negative_sample,
                              thirdserver, args.kg_hop,args.GNNhop,args.sampleradio))
    # build server
    server = server(user_list, user_batch, user_id_list, item_id_list,
                    embed_size, lr, args.neighbor_num,
                    n_entity, n_relation, triples, args.l2_weight,args.kg_hop,args.GNNhop)
    count = 0

    # train and evaluate
    acc_best = 0
    auc_test, f1_test, acc_test = 0, 0, 0
    epoch = 0
    recall_best = 0
    recall_test, ndcg_test, precision_test = 0,0,0
    print("固定客户端上面的图")
    uids_items = server.gen_uid_item(user_list)
    for user in tqdm(user_list):
        user.send_items(uids_items)
    while 1:
        print("服务器开始训练{}轮".format(epoch))
        epoch += 1
        for i in range(args.valid_step):
            # 服务器开始训练了
            server.train()

        trainauc, trainf1, trainacc = ctr_eval(server, train_data_test)
        print('train auc: {}, traind f1:{},train acc:{}'.format(trainauc, trainf1, trainacc))

        auc, f1, acc = ctr_eval(server, valid_data)
        print('valid auc: {}, valid f1:{},valid acc:{}'.format(auc, f1, acc))
        auc_test_, f1_test_, acc_test_ = ctr_eval(server, test_data)
        print('test_auc_: {}, test_f1_:{},test_acc_:{}'.format(auc_test_, f1_test_, acc_test_))
        if acc > acc_best:
            acc_best = acc
            count = 0
            auc_test, f1_test, acc_test = ctr_eval(server, test_data)
        else:
            count += 1
        if count == args.lr_step:
            server.reduce_LR(index=0.5)
        if count > args.stop_step:
            print('not improved for {} epochs, stop trianing'.format(args.stop_step))
            print('final-test auc: {}, test f1:{},test acc:{}'.format(auc_test, f1_test, acc_test))
            break
