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
from sys import getsizeof

import os
import sys
sys.path.append('..')
sys.path.append('D:\\Recdots\\Recdots-master\\rec')

from rec_app import log
import logging

from collections import Counter
import csv
from itertools import islice
import pandas as pd
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def topK_scores(test, predict, topk, user_count, item_count):

    PrecisionSum = np.zeros(topk+1)
    RecallSum = np.zeros(topk+1)
    F1Sum = np.zeros(topk+1)
    NDCGSum = np.zeros(topk+1)
    OneCallSum = np.zeros(topk+1)
    DCGbest = np.zeros(topk+1)
    MRRSum = 0
    MAPSum = 0
    total_test_data_count = 0
    for k in range(1, topk+1):
        DCGbest[k] = DCGbest[k - 1]
        DCGbest[k] += 1.0 / math.log(k + 1)
    for i in range(user_count):
        user_test = []
        user_predict = []
        test_data_size = 0
        for j in range(item_count):
            if test[i * item_count + j] == 1.0:
                test_data_size += 1
            user_test.append(test[i * item_count + j])
            user_predict.append(predict[i * item_count + j])
        if test_data_size == 0:
            continue
        else:
            total_test_data_count += 1
        predict_max_num_index_list = map(user_predict.index, heapq.nlargest(topk, user_predict))
        predict_max_num_index_list = list(predict_max_num_index_list)
        hit_sum = 0
        DCG = np.zeros(topk + 1)
        DCGbest2 = np.zeros(topk + 1)
        for k in range(1, topk + 1):
            DCG[k] = DCG[k - 1]
            item_id = predict_max_num_index_list[k - 1]
            if user_test[item_id] == 1:
                hit_sum += 1
                DCG[k] += 1 / math.log(k + 1)
            # precision, recall, F1, 1-call
            prec = float(hit_sum / k)
            rec = float(hit_sum / test_data_size)
            f1 = 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            PrecisionSum[k] += float(prec)
            RecallSum[k] += float(rec)
            F1Sum[k] += float(f1)
            if test_data_size >= k:
                DCGbest2[k] = DCGbest[k]
            else:
                DCGbest2[k] = DCGbest2[k-1]
            NDCGSum[k] += DCG[k] / DCGbest2[k]
            if hit_sum > 0:
                OneCallSum[k] += 1
            else:
                OneCallSum[k] += 0
        # MRR
        p = 1
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                break
            p += 1
        MRRSum += 1 / float(p)
        # MAP
        p = 1
        AP = 0.0
        hit_before = 0
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                AP += 1 / float(p) * (hit_before + 1)
                hit_before += 1
            p += 1
        MAPSum += AP / test_data_size
        if total_test_data_count != 0:
            print('MAP:', MAPSum / total_test_data_count)
            print('MRR:', MRRSum / total_test_data_count)
            print('Prec@5:', PrecisionSum[4] / total_test_data_count)
            print('rec_app@5:', RecallSum[4] / total_test_data_count)
            print('F1@5:', F1Sum[4] / total_test_data_count)
            print('NDCG@5:', NDCGSum[4] / total_test_data_count)
            print('1-call@5:', OneCallSum[4] / total_test_data_count)

    return


'''
函数说明:BPR类（包含所需的各种参数）
Parameters:
    无
Returns:
    无
'''
class BPR():

    # 用户数
    user_count = 459
    # 物品数
    item_count = 33980
    # k个主题，k数
    latent_factors = 20
    # 步长α
    lr = 0.01
    # 参数λ
    reg = 0.01
    # 训练次数
    train_count = 1000

    # 训练集路径
    train_data_path = 'train.txt'
    # train_data_path = 'train.csv'
    # 测试集路径
    test_data_path = 'test.txt'
    # test_data_path = 'test.csv'
    # U-I的大小
    size_u_i = user_count * item_count

    # 生成一个用户数*项目数大小的全0矩阵 user_count行，item_count列的列表
    test_data = np.zeros((user_count, item_count))
    print("test_data_type", type(test_data))
    # 生成一个一维的全0矩阵
    test = np.zeros(size_u_i)
    # 再生成一个一维的全0矩阵
    predict_ = np.zeros(size_u_i)

    def __init__(self, hash_user, user_hash, hash_item, item_hash, user_ratings, U, V, biasV, uid_clicked):
        self.hash_user = hash_user
        self.user_hash = user_hash
        self.hash_item = hash_item
        self.item_hash = item_hash
        self.user_ratings = user_ratings
        self.U = U
        self.V = V
        self.biasV = biasV
        self.uid_clicked = uid_clicked
        # # hash映射
        # hash_user = dict()  # hash映射 0開始的id：本身id
        # user_hash = dict()  # hash映射 本身id：0開始的id
        # hash_item = dict()  # hash映射 0開始的id：本身id
        # item_hash = dict()  # hash映射 本身id：0開始的id
        # user_ratings = defaultdict(set)  # 自定义字典的value为集合（set）  例[('blue', {2, 4}), ('red', {1, 3})]
        # # 随机设定的U，V矩阵(即公式中的Wuk和Hik)矩阵     Xui = W*HT
        # U = np.random.rand(user_count,
        #                    latent_factors) * 0.01  # 大小无所谓  np.random.rand返回一组user_count行 latent_factors列的（0,1]的随机数列表
        # V = np.random.rand(item_count, latent_factors) * 0.01
        # biasV = np.random.rand(item_count) * 0.01


    # 获取U-I数据对应
    '''
    函数说明：通过文件路径，获取U-I数据
    Paramaters:
        输入要读入的文件路径path
    Returns:
        输出一个字典user_ratings，包含用户-项目的键值对
    '''
    def load_data(self, path):
        logger = logging.getLogger()
        logs = log.CommonLog(logger, "D:\\Recdots\\Recdots-master\\rec\\rec_app\\rec_log\\load_data.log")
        logs.info('Launch the Django service and start load data!')



        # 读用户表
        start_load_user = time.time()

        user_set = set()

        # for count in range(4000):
        #     select_sql = 'SELECT * FROM user ORDER BY create_time DESC LIMIT 1000 OFFSET %d' % (count * 1000)
        #     select_result = select_db(select_sql)
        #     for j in range(1000):
        #         try:
        #             u = int(select_result[j]['user_id'])
        #             user_set.add(u)
        #             print(u)
        #         except:
        #             print("第" + str(count * 1000 + j) + "条数据读取报错")
        #             # break
        #     print("length of user : " + str(len(user_set)))

        with open('user.csv', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            print(reader)
            for line in reader:
                u = int(line[0])
                user_set.add(u)
            # print(user_set)
            print("length of user :"+str(len(user_set)))
        end_load_user = time.time()
        duration = str(round((end_load_user - start_load_user), 2))
        logs.info('Finish loading user. Total time : ' + str(duration) + ' s.\n' +
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------')



        # 读物品表
        start_load_item = time.time()

        item_set = set()

        for count in range(40):
            select_sql = 'SELECT * FROM item ORDER BY create_time DESC LIMIT 1000 OFFSET %d' % (count * 1000)
            select_result = select_db(select_sql)
            for j in range(1000):
                try:
                    u = int(select_result[j]['item_id'])
                    item_set.add(u)
                    print(u)
                except:
                    print("第" + str(count * 1000 + j) + "条数据读取报错")
                    #break
            print("length of item : " + str(len(item_set)))

        print("length of item : " + str(len(item_set)))
        # with open('item.csv', encoding='utf-8') as fp:
        #     reader = csv.reader(fp)
        #
        #     # for line in reader:
        #     for line in islice(reader, 1, None):
        #         i = int(line[0])
        #         item_set.add(i)
        #     # print(item_set)
        #     print("length of item :"+str(len(item_set)))
        end_load_item = time.time()
        duration = str(round((end_load_item - start_load_item), 2))
        logs.info('Finish loading item. Total time : ' + str(duration) + ' s.\n' +
                  '---------------------------------------------------------------------------------------------------------------\n' +
                  '---------------------------------------------------------------------------------------------------------------\n' +
                  '---------------------------------------------------------------------------------------------------------------')

        # 读交互表
        with open('user_item.csv', encoding='utf-8') as fp:
            reader = csv.reader(fp)

            # for line in reader:
            for line in islice(reader, 1, None):
                u = int(line[0])
                i = int(line[1])
                user_set.add(u)
                item_set.add(i)
            print("length of user :"+str(len(user_set)))
            print("length of item :"+str(len(item_set)))

        # 建立hash映射
        index1 = 0
        for user in user_set:
            self.user_hash[user] = index1
            self.hash_user[index1] = user
            index1 += 1
        print("the size of user_hash: "+str(index1))
        index2 = 0
        for item in item_set:
            self.item_hash[item] = index2
            self.hash_item[index2] = item
            index2 += 1
        print("the size of item_hash: "+str(index2))

        # 建立user_ratings字典
        start_load_user_item = time.time()

        filename = 'D:\\Recdots\\Recdots-master\\rec\\rec_app\\user_item.csv'
        total_lines = sum(1 for line in open(filename, encoding='utf-8'))
        print('The total lines is ', total_lines)
        # 读取
        csv_result = pd.read_csv('D:\\Recdots\\Recdots-master\\rec\\rec_app\\user_item.csv')
        if total_lines >= 1000000:
            start_line = total_lines - 1000000
            end_line = start_line + 1000
        else:
            start_line = 0
            end_line = start_line + 1000

        for i in range(1000):

            csv_result1 = csv_result.iloc[start_line:end_line, 0:2]
            row_list = csv_result1.values.tolist()
            print(f"行读取结果：{row_list}")
            for j in range(1000):
                try:
                    user_set.add(row_list[j][0])
                    item_set.add(row_list[j][1])
                    u = self.user_hash[row_list[j][0]]
                    i = self.item_hash[row_list[j][1]]
                    self.user_ratings[u].add(i)
                except:
                    print(str(j + i * 1000) + "行数据 bug or null")

            start_line += 1000
            end_line = start_line + 1000

        # with open('user_item.csv', encoding='utf-8') as fp:
        #     reader2 = csv.reader(fp)
        #     # 遍历数据
        #     index = 0
        #     for line in islice(reader2, 1, None):
        #         user = int(line[0])
        #         item = int(line[1])
        #         u = self.user_hash[user]
        #         i = self.item_hash[item]
        #         self.user_ratings[u].add(i)
        #         index += 1
        #     print("the size of behavior: "+str(index))

        end_load_user_item = time.time()
        duration = str(round((end_load_user_item - start_load_user_item), 2))
        logs.info('Finish loading user_item. Total time : ' + str(duration) + ' s.\n' +
                  '---------------------------------------------------------------------------------------------------------------\n' +
                  '---------------------------------------------------------------------------------------------------------------\n' +
                  '---------------------------------------------------------------------------------------------------------------')


        # return self.user_ratings
        # import pdb;pdb.set_trace()

        '''
        函数说明：通过文件路径，获取测试集数据
        Paramaters：
            测试集文件路径path
        Returns:
            输出一个numpy.ndarray文件（n维数组）test_data,其中把含有反馈信息的数据置为1
        '''
        # 获取测试集的评分矩阵
    def load_test_data(self, path):
        # file = open('test1.txt', 'r')
        # for line in file:
        #     # try:
        #         line = line.split('\t')
        #         user = int(line[0])
        #         item = int(line[1])
        #         u = self.user_hash[user]
        #         i = self.item_hash[item]
        #         # self.test_data[u - 1][i - 1] = 1       #???
        #         self.test_data[u][i] = 1  # ???
        # # except:
        #     #     print('第{0}条数据处理失败'.format(line))

        select_sql = 'SELECT * FROM behavior ORDER BY create_time DESC limit 200'
        select_result = select_db(select_sql)
        for i in range(200):
            user = select_result[i]['user_id']
            item = select_result[i]['item_id']
            u = self.user_hash[user]
            i = self.item_hash[item]
            self.test_data[u][i] = 1

        '''
        函数说明：对训练集数据字典处理，通过随机选取，（用户，交互，为交互）三元组，更新分解后的两个矩阵
        Parameters：
            输入要处理的训练集用户项目字典
        Returns：
            对分解后的两个矩阵以及偏置矩阵分别更新
        '''
    def train(self):
        for user in range(self.user_count):
            # 随机获取一个用户
            # u = random.randint(1, self.user_count) # 找到一个user
            u = random.randint(0, self.user_count - 1) # 找到一个user
            # 训练集和测试集不是全都一样的,比如train有948,而test最大为943
            if u not in self.user_ratings.keys():
                continue
            # 从用户的U-I中随机选取1个Item
            i = random.sample(self.user_ratings[u], 1)[0] # 找到一个item，被评分 random.sample()截取列表user_ratings_train[u]指定长度1的随机数产生新列表
            # 随机选取一个用户u没有评分的项目
            # j = random.randint(1, self.item_count)
            j = random.randint(0, self.item_count - 1)
            while j in self.user_ratings[u]:
                # j = random.randint(1, self.item_count) # 找到一个item，没有被评分
                j = random.randint(0, self.item_count - 1)  # 找到一个item，没有被评分
            # 构成一个三元组（user:u,item_have_score:i,item_no_score:j)
            # # python中的取值从0开始
            # u = u - 1
            # i = i - 1
            # j = j - 1
            #BPR
            # try:
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]  # np.dot表示向量内积或矩阵乘法 Xui=WU*Hi=Wuf*Hif(f在1-k的累加)
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # print(loss_func)
            # except:
            #     continue

            # 更新这个u和v矩阵
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # 更新偏置项
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])

    def online_train(self, user, item):
            # 获取online_train开始时间
            start_online_train = time.time()

            logger = logging.getLogger()
            logs = log.CommonLog(logger, "rec_app/rec_log/online_tarin.log")
            logs.info('Start online training!\n')

            u = self.user_hash[user]
            i = self.item_hash[item]
            self.user_ratings[u].add(i)
            self.uid_clicked[user].add(item)
            # 随机选取一个用户u没有评分的项目
            j = random.randint(0, self.item_count - 1)
            while j in self.user_ratings[u]:
                j = random.randint(0, self.item_count - 1) # 找到一个item，没有被评分
            # 构成一个三元组（user:u,item_have_score:i,item_no_score:j)
            # # python中的取值从0开始
            # u = u - 1
            # i = i - 1
            # j = j - 1
            #BPR
            try:
                r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]  # np.dot表示向量内积或矩阵乘法 Xui=WU*Hi=Wuf*Hif(f在1-k的累加)
                r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
                r_uij = r_ui - r_uj
                loss_func = -1.0 / (1 + np.exp(r_uij))
            except:
                print("error")

            # 更新这个u和v矩阵
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # 更新偏置项
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])



            predict = np.mat(self.U[u]) * np.mat(self.V.T)
            # 求item的top-K的索引
            topK_matrix = partition_arg_topK(predict, 50, axis=1)
            topK_matrix1 = np.array(topK_matrix)
            print(topK_matrix1[0])

            # 开始查询
            start_select = time.time()



            end_select = time.time()
            select_duration = str(round((end_select - start_select) * 1000, 2))
            logs.info('Query the user whose user ID is ' + str(user) + ' from the database. Using time ' + str(
                select_duration) + ' ms.')

            # 获得包含所有user的点击set
            clicked_set = set()
            select_sql = 'SELECT * FROM rec_his WHERE clicked_ids IS NOT NULL'
            select_result = select_db(select_sql)
            for i in range(len(select_result)):
                uid = select_result[i]['user_id']
                itemid = select_result[i]['clicked_ids']
                self.uid_clicked[int(uid)].add(int(itemid))
            for uid in self.uid_clicked.keys():
                for i in self.uid_clicked[uid]:
                    clicked_set.add(i)
            print(clicked_set)

            # 热门过滤
            item_set = set()
            hot_set = hot_rec()
            print(hot_set)
            index = 0
            for i in hot_set:
                if i not in clicked_set:
                    item_set.add(i)
                    index += 1

                if index >= 21:
                    break
                print(i)
            print(item_set)  # item_set保留21个热门


            select_sql = 'SELECT * FROM rec WHERE user_id="%d"'% user  # %s（）或者用format
            select_result = select_db(select_sql)
            index2 = select_result[0]['index2']
            select_lastsql = 'SELECT * FROM rec_his ORDER BY create_time DESC limit 1'
            select_last = select_db(select_lastsql)
            id2 = select_last[0]['id'] + 1

            # bpr过滤
            count = 0
            for item_index in range(len(topK_matrix1[0])):
                m = topK_matrix1[0][item_index]
                # topK_matrix1[x1][y1] = self.hash_item[topK_matrix1[x1][y1]]
                item_id = int(self.hash_item[m])
                if (item_id not in item_set) and (item_id not in self.uid_clicked[user]):
                    item_set.add(item_id)
                    count += 1

                if count >= 29:
                    break

            # item_ids字符串构建
            item_ids = ''
            for item_id in item_set:
                item_ids = item_ids + str(item_id) + ','
            item_ids = item_ids[:-1]

            user_id = user
            # update一下rec
            start_update = time.time()
            update_sql = 'update rec set item_ids = "%s" where index2 = "%d"' % (item_ids, index2)
            update_db(update_sql)

            # update rec_his
            insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids, clicked_ids) VALUES(%d, %d, '%s', %d)" % (
            id2, user_id, item_ids, int(item))
            print(insert_sql2)
            print(item)
            insert_db(insert_sql2)

            end_update = time.time()
            update_duration = str(round((end_update - start_update) * 1000, 2))

            logs.info('Insert a row of data. Using time ' + str(update_duration) + ' ms.')

            # 末尾追加写入，文件必须已存在
            with open('D:\\Recdots\\Recdots-master\\rec\\rec_app\\user_item.csv', mode='a', newline='',
                      encoding='utf8') as cfa:
                wf = csv.writer(cfa)
                now = datetime.datetime.now()
                # data2 = [[u1, u2, u3], ['小力', '12', '男'], ]
                # for i in data2:
                #     wf.writerow(i)
                data2 = [user, item, now]
                wf.writerow(data2)


            end_online_train = time.time()
            online_train_duration = str(round((end_online_train - start_online_train) * 1000, 2))
            logs.info('Finish online train. Total time: ' + str(online_train_duration) + ' ms.\n' +
                      '---------------------------------------------------------------------------------------------------------------\n' +
                      '---------------------------------------------------------------------------------------------------------------\n' +
                      '---------------------------------------------------------------------------------------------------------------')

    '''
    函数说明：通过输入分解后的用户项目矩阵得到预测矩阵predict
    Parameters:
        输入分别后的用户项目矩阵
    Returns：
        输出相乘后的预测矩阵，即我们所要的评分矩阵
    '''
    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)    #??????
        return predict

    def pre_handel(self, predict, item_count):
        # Ensure the recommendation cannot be positive items in the training set.
        for u in self.user_ratings.keys():
            for j in self.user_ratings[u]:
                try:
                    predict[(int(u) - 1) * int(item_count) + int(j) - 1] = 0
                    # print("pre_handel失败")
                except:
                    continue

        return predict



    #??????
    def main(self):

        logger = logging.getLogger()
        logs = log.CommonLog(logger, "D:\\Recdots\\Recdots-master\\rec\\rec_app\\rec_log\\offline_tarin.log")
        logs.info('Launch the Django service and start offline training!')
        # 获取offline_train开始时间
        start_offline_train = time.time()

        # 获取U-I的{1:{2,5,1,2}....}数据
        # user_ratings_train = self.load_data(self.train_data_path)
        self.load_data(self.train_data_path)

        # print(user_ratings_train)
        # 获取测试集的评分矩阵
        self.load_test_data(self.test_data_path)
        # 将test_data矩阵拍平
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0




        # 离线训练
        for i in range(self.train_count):
            start_time = time.time()

            self.train()  # 训练train_count次完成

            end_time = time.time()
            # 计算每一轮offline_train的时长
            each_duration = str(round((end_time - start_time) * 1000, 2))
            logs.info('Finish the ' + str(i) + ' th offline training. Using time ' + each_duration + ' ms.')



        print("size of hash_user is: " + str(round(getsizeof(self.hash_user) / (1024 ** 2), 2)) + "MB")
        print("size of user_hash is: " + str(round(getsizeof(self.user_hash) / (1024 ** 2), 2)) + "MB")
        print("size of hash_item is: " + str(round(getsizeof(self.hash_item) / (1024 ** 2), 2)) + "MB")
        print("size of item_hash is: " + str(round(getsizeof(self.item_hash) / (1024 ** 2), 2)) + "MB")
        print("size of user_ratings is: " + str(round(getsizeof(self.user_ratings) / (1024 ** 2), 2)) + "MB")
        print("size of U is: "+str(round(getsizeof(self.U) / (1024 ** 2), 2))+"MB")
        print("size of V is: " + str(round(getsizeof(self.V) / (1024 ** 2), 2)) + "MB")
        print("size of biasV is: "+str(round(getsizeof(self.biasV) / (1024 ** 2), 2))+"MB")
        print("size of uid_clicked is: " + str(round(getsizeof(self.uid_clicked) / (1024 ** 2), 2)) + "MB")


        end_offline_train = time.time()
        duration = str(round((end_offline_train - start_offline_train), 2))
        logs.info('Finish offline training. Total time : ' + str(duration) + ' s.\n' +
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------')

        start_offline_insert = time.time()



        # predict_matrix = self.predict(self.U, self.V) #??????????????
        # print(predict_matrix)

        # 获得包含所有user的点击set
        clicked_set = set()
        select_sql = 'SELECT * FROM rec_his WHERE clicked_ids IS NOT NULL'
        select_result = select_db(select_sql)
        for i in range(len(select_result)):
            uid = select_result[i]['user_id']
            itemid = select_result[i]['clicked_ids']
            self.uid_clicked[int(uid)].add(int(itemid))
        for uid in self.uid_clicked.keys():
            for i in self.uid_clicked[uid]:
                clicked_set.add(i)
        print(clicked_set)

        # 热门过滤
        item_set = set()
        hot_set = hot_rec()
        print(hot_set)
        index = 0
        for i in hot_set:
            if i not in clicked_set:
                item_set.add(i)
                index += 1

            if index >= 21:
                break
            print(i)
        print(item_set)  # item_set保留21个热门


        id = 0
        select_lastsql = 'SELECT * FROM rec_his ORDER BY create_time DESC limit 1'
        select_last = select_db(select_lastsql)
        id2 = select_last[0]['id'] + 1

        for user_index in range(459):

            predict = np.mat(self.U[user_index]) * np.mat(self.V.T)
            print(predict)
            print(round(getsizeof(predict) / (1024 ** 2), 2))
            # 求item的top-K的索引
            topK_matrix = partition_arg_topK(predict, 50, axis=1)    #top50
            topK_matrix1 = np.array(topK_matrix)
            print(topK_matrix1[0])
            user_id = int(self.hash_user[user_index])


            # bpr过滤
            count = 0
            for item_index in range(len(topK_matrix1[0])):
                index2 = id
                m = topK_matrix1[0][item_index]
                # topK_matrix1[x1][y1] = self.hash_item[topK_matrix1[x1][y1]]
                item_id = int(self.hash_item[m])
                if (item_id not in item_set) and (item_id not in self.uid_clicked[user_id]):
                    item_set.add(item_id)
                    count += 1

                if count >= 29:
                    break

                # # insert_sql1 = "INSERT INTO rec(index2, user_id, item_ids, create_time) VALUES(%d, %d, %d, str_to_date(%s,'%%Y-%%m-%%d %%H:%%M:%%S'))" % (id, user_id, item_ids, now_time)

                # insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids) VALUES(%d, %d, %d)" % (id, user_id, item_ids)
                # # insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids, create_time) VALUES(%d, %d, %d, str_to_date(%s,'%%Y-%%m-%%d %%H:%%M:%%S'))" % (id, user_id, item_ids, now_time)
                # insert_db(insert_sql2)


            # item_ids字符串构建
            item_ids = ''
            for item_id in item_set:
                item_ids = item_ids + str(item_id) + ','
            item_ids = item_ids[:-1]
            print(user_id, item_ids)


            select_sql = 'SELECT * FROM rec WHERE index2="%d"' % id  # %s（）或者用format
            select_result = select_db(select_sql)
            # insert数据库表rec,rec_his
            if (select_result == ()):
                # insert
                insert_sql1 = "INSERT INTO rec(index2, user_id, item_ids) VALUES(%d, %d, '%s')" % (
                id, user_id, item_ids)
                insert_db(insert_sql1)
            else:
                # update
                update_sql = 'update rec set user_id = "%d", item_ids = "%s" where index2 = "%d"' % (user_id, item_ids, id)
                update_db(update_sql)

            insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids) VALUES(%d, %d, '%s')" % (id2, user_id, item_ids)
            insert_db(insert_sql2)

            id += 1
            id2 += 1


        # for x1 in range(len(topK_matrix1)):
        #     user_id = int(self.hash_user[x1])
        #     select_sql = 'SELECT * FROM rec WHERE user_id="%d"' % user_id  # %s（）或者用format
        #     select_result = select_db(select_sql)
        #     item_ids = ''
        #     for y1 in range(len(topK_matrix1[x1])):
        #         index2 =id
        #         m = topK_matrix1[x1][y1]
        #         # topK_matrix1[x1][y1] = self.hash_item[topK_matrix1[x1][y1]]
        #         item_id = int(self.hash_item[m])
        #         item_ids = item_ids + str(item_id) + ','
        #         # # insert_sql1 = "INSERT INTO rec(index2, user_id, item_ids, create_time) VALUES(%d, %d, %d, str_to_date(%s,'%%Y-%%m-%%d %%H:%%M:%%S'))" % (id, user_id, item_ids, now_time)
        #
        #         # insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids) VALUES(%d, %d, %d)" % (id, user_id, item_ids)
        #         # # insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids, create_time) VALUES(%d, %d, %d, str_to_date(%s,'%%Y-%%m-%%d %%H:%%M:%%S'))" % (id, user_id, item_ids, now_time)
        #         # insert_db(insert_sql2)
        #
        #     item_ids = item_ids[:-1]
        #     print(user_id, item_ids)
        #     if (select_result == ()):
        #         # insert
        #         insert_sql1 = "INSERT INTO rec(index2, user_id, item_ids) VALUES(%d, %d, '%s')" % (id, user_id, item_ids)
        #         insert_db(insert_sql1)
        #     else:
        #         # update
        #         update_sql = 'update rec set item_ids = "%s" where index2 = "%d"' % (item_ids, id)
        #         update_db(update_sql)
        #
        #     insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids) VALUES(%d, %d, '%s')" % (id2, user_id, item_ids)
        #     insert_db(insert_sql2)
        #
        #     id += 1
        #     id2 += 1


        end_offline_insert = time.time()
        duration = str(round((end_offline_insert - start_offline_insert), 2))
        logs.info('Finish offline inserting rec and rec_his. Total time : ' + str(duration) + ' s.\n' +
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------')

        # topK_matrix1 = naive_arg_topK(predict_matrix, 10, axis=1)
        # print(topK_matrix1)

        # # 在线训练
        # #self.online_train(self, user, item)
        # # old_time1 = '2022-10-11 19:16:32'
        # # old_time = datetime.datetime.strptime(old_time1, "%Y-%m-%d %H:%M:%S")
        # select_sql = 'SELECT * FROM behavior ORDER BY create_time DESC limit 1'  # %s（）或者用format
        # data = select_db(select_sql)
        # old_time = data[0]['create_time']
        # while 1:
        #     select_sql = 'SELECT * FROM behavior ORDER BY create_time DESC limit 1'  # %s（）或者用format
        #     m = select_db(select_sql)
        #
        #     uid = m[0]['user_id']
        #     iid = m[0]['item_id']
        #     u = self.user_hash[uid]
        #     i = self.item_hash[iid]
        #     # print(m)
        #     # print(m[0]['user_id'])
        #     # print(m[0]['item_id'])
        #     # print(m[0]['create_time'])
        #
        #     if m[0]['create_time'] > old_time:
        #         self.online_train(uid, iid)
        #         predict_matrix = self.predict(self.U, self.V)
        #         print(predict_matrix)
        #         topK_matrix = partition_arg_topK(predict_matrix, 10, axis=1)
        #         topK_matrix1 = np.array(topK_matrix)
        #
        #         select_sql = 'SELECT * FROM rec WHERE user_id="%d"'% uid  # %s（）或者用format
        #         m1 = select_db(select_sql)
        #         index2 = m1[0]['index2']
        #         for y in range(len(topK_matrix1[u])):
        #             ii = topK_matrix1[u][y]
        #             # topK_matrix1[x1][y1] = self.hash_item[topK_matrix1[x1][y1]]
        #             user_id = int(self.hash_user[u])
        #             item_ids = int(self.hash_item[ii])
        #             # update一下rec
        #             # update_sql = 'update rec set item_ids = "%d" where index2 = "%d"' % (item_ids, index2)
        #             # update_db(update_sql)
        #             index2 += 1
        #
        #         old_time = m[0]['create_time']


            # print(old_time > m[0]['create_time'])




        # self.predict_ = predict_matrix.getA().reshape(-1)  # .getA()将自身矩阵变量matrix转化为nd array类型的变量 reshape(-1)#改成一串，没有行列的列表
        # # import pdb; pdb.set_trace()
        # print("predict_new", self.predict_)
        # self.predict_ = self.pre_handel(self.predict_, self.item_count)
        # auc_score = roc_auc_score(self.test, self.predict_)
        # print('AUC:', auc_score)
        # Top-K evaluation
        # topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)

        end_offline = time.time()
        duration = str(round((end_offline - start_offline_train), 2))
        logs.info('Finish offline inserting rec and rec_his. Total time : ' + str(duration) + ' s.\n' +
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------')





    # ???????????????????????????????????????????????????????????????в????????????????????
    # Paramaters:
    #     ????????????伯??????????????????????
    # Returns:
    #     ???????????????????????????


def naive_arg_topK(matrix, K, axis=0):
        """
        perform topK based on np.argsort
        :param matrix: to be sorted
        :param K: select and sort the top K items
        :param axis: dimension to be sorted.
        :return:
        """
        full_sort = np.argsort(matrix, axis=axis)
        return full_sort.take(np.arange(K), axis=axis)

def partition_arg_topK(matrix, K, axis=0):
        """
        perform topK based on np.argpartition
        :param matrix: to be sorted
        :param K: select and sort the top K items
        :param axis: 0 or 1. dimension to be sorted.
        :return:
        """
        a_part = np.argpartition(matrix, K, axis=axis)
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis])
            a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
            return a_part[0:K, :][a_sec_argsort_K, row_index]
        else:
            column_index = np.arange(matrix.shape[1 - axis])[:, None]
            a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
            return a_part[:, 0:K][column_index, a_sec_argsort_K]

def insert_db(insert_sql):
    """插入"""
    # 建立数据库连接
    db = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        passwd="123456",
        db="recdots"
    )
    # 通过 cursor() 创建游标对象
    cur = db.cursor()
    try:
        # 使用 execute() 执行sql
        cur.execute(insert_sql)
        # 提交事务
        db.commit()
    except Exception as e:
        print("操作出现错误：{}".format(e))
        # 回滚所有更改
        db.rollback()
    finally:
        # 关闭游标
        cur.close()
        # 关闭数据库连接
        db.close()

def select_db(select_sql):
    """查询"""
    # 建立数据库连接
    db = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        passwd="123456",
        db="recdots"
    )
    # 通过 cursor() 创建游标对象，并让查询结果以字典格式输出
    cur = db.cursor(cursor=pymysql.cursors.DictCursor)
    # 使用 execute() 执行sql
    cur.execute(select_sql)
    # 使用 fetchall() 获取所有查询结果
    data = cur.fetchall()
    # 关闭游标
    cur.close()
    # 关闭数据库连接
    db.close()
    return data

def update_db(update_sql):
        ''' 更新数据库操作 '''

        db = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            passwd="123456",
            db="recdots"
        )

        cursor = db.cursor()

        try:
            # 执行sql
            cursor.execute(update_sql)
            # tt = self.cursor.execute(sql) # 返回 更新数据 条数 可以根据 返回值 判定处理结果
            # print(tt)
            db.commit()
        except:
            # 发生错误时回滚
            db.rollback()
        finally:
            cursor.close()

def hot_rec():

    item_list = []
    with open('user_item.csv', encoding='utf-8') as fp:
        reader = csv.reader(fp)

        # for line in reader:
        for line in islice(reader, 1, None):
            i = int(line[1])
            item_list.append(i)
        print("length of item :" + str(len(item_list)))


    collection_items = Counter(item_list)
    # print(collection_items)
    # 还可以输出频率最大的n个元素,类型为list
    most_counterNum = collection_items.most_common(50)
    print(most_counterNum)
    print(most_counterNum[0])  #元组(id，count)
    print(most_counterNum[0][0])  #id
    print(type(most_counterNum))

    hot_set = set()
    for i in range(50):
        hot_set.add(most_counterNum[i][0])

    print(hot_set)
    return hot_set

if __name__ == '__main__':
    # 调用类的主函数

    # with open('t.csv', encoding='utf-8') as fp:
    #     reader = csv.reader(fp)
    #     # # 获取标题
    #     # header = next(reader)
    #     # print(header)
    #     # 遍历数据
    #     for i in reader:
    #         print(i)
    print(time.time())
    bpr = BPR()
    bpr.main()
    print(time.time())
