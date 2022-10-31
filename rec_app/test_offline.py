# coding=utf-8
import random
from collections import defaultdict
# from sklearn.metrics import roc_auc_score
import heapq
import numpy as np
import math
from itertools import islice
import time
import datetime

# ???????top_K????
# import pandas
import csv
# import pymysql
import pymysql

import sys
sys.path.append('..')
sys.path.append('D:\\Recdots\\Recdots-master\\rec')

from offline_online import BPR


if __name__ == '__main__':
    # 调用类的主函数

    # 用户数
    user_count = 459
    # 物品数
    item_count = 33980
    # k个主题，k数
    latent_factors = 20

    # hash映射
    hash_user = dict()  # hash映射 0開始的id：本身id
    user_hash = dict()  # hash映射 本身id：0開始的id
    hash_item = dict()  # hash映射 0開始的id：本身id
    item_hash = dict()  # hash映射 本身id：0開始的id
    user_ratings = defaultdict(set)  # 自定义字典的value为集合（set）  例[('blue', {2, 4}), ('red', {1, 3})]
    uid_clicked = defaultdict(set)  # 自定义字典的value为集合（set） [(uid,{clicked_iid1,clicked_iid2...}),(uid,{clicked_iid1,clicked_iid2...})]
    # 随机设定的U，V矩阵(即公式中的Wuk和Hik)矩阵     Xui = W*HT
    U = np.random.rand(user_count,
                       latent_factors) * 0.01  # 大小无所谓  np.random.rand返回一组user_count行 latent_factors列的（0,1]的随机数列表
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01


    print(time.ctime())

    global bpr
    bpr = BPR(hash_user, user_hash, hash_item, item_hash, user_ratings, U, V, biasV, uid_clicked)
    bpr.main()
    user = 4128772
    item = '55770/2'
    bpr.online_train(user, item)

    # user = 4128772
    # item = 73633
    # bpr.online_train(user, item)


    print(time.ctime())

