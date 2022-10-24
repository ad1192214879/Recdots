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

from rec_app import log
import logging

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
    user_count = 453
    # 物品数
    item_count = 23686
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

        with open('user_item.csv', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            user_set = set()
            item_set = set()
            # for line in reader:
            for line in islice(reader, 1, None):
                u = int(line[0])
                i = int(line[1])
                user_set.add(u)
                item_set.add(i)
            print(len(user_set))
            print(len(item_set))

            index1 = 0
            for user in user_set:
                self.user_hash[user] = index1
                self.hash_user[index1] = user
                index1 += 1
            print(index1)
            index2 = 0
            for item in item_set:
                self.item_hash[item] = index2
                self.hash_item[index2] = item
                index2 += 1
            print(index2)

        with open('user_item.csv', encoding='utf-8') as fp:
            reader2 = csv.reader(fp)
            # 遍历数据
            index = 0
            for line in islice(reader2, 1, None):
                user = int(line[0])
                item = int(line[1])
                # self.user_hash[u] = index
                # self.hash_user[index] = u
                # self.item_hash[i] = index
                # self.hash_item[index] = i
                u = self.user_hash[user]
                i = self.item_hash[item]
                self.user_ratings[u].add(i)
                index += 1
            print(index)

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

            predict_matrix = self.predict(self.U, self.V)
            print(predict_matrix)
            topK_matrix = partition_arg_topK(predict_matrix, 10, axis=1)
            topK_matrix1 = np.array(topK_matrix)

            # 开始查询
            start_select = time.time()

            select_sql = 'SELECT * FROM rec WHERE user_id="%d"'% user  # %s（）或者用format
            select_result = select_db(select_sql)

            end_select = time.time()
            select_duration = str(round((end_select - start_select) * 1000, 2))
            logs.info('Query the user whose user ID is ' + str(user) + ' from the database. Using time ' + str(
                select_duration) + ' ms.')

            index2 = select_result[0]['index2']
            select_lastsql = 'SELECT * FROM rec_his ORDER BY create_time DESC limit 1'
            select_last = select_db(select_lastsql)
            id2 = select_last[0]['id'] + 1
            item_ids = ''
            user_id = user
            for y in range(10):
                ii = topK_matrix1[u][y]
                # topK_matrix1[x1][y1] = self.hash_item[topK_matrix1[x1][y1]]

                item_id = int(self.hash_item[ii])
                item_ids = item_ids + str(item_id) + ','

            item_ids = item_ids[:-1]
            print(item_ids)
            # update一下rec
            start_update = time.time()
            update_sql = 'update rec set item_ids = "%s" where index2 = "%d"' % (item_ids, index2)
            update_db(update_sql)

            # update rec_his
            insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids, clicked_ids) VALUES(%d, %d, '%s', %d)" % (
            id2, user_id, item_ids, int(item))
            insert_db(insert_sql2)

            end_update = time.time()
            update_duration = str(round((end_update - start_update) * 1000, 2))

            logs.info('Insert a row of data. Using time ' + str(update_duration) + ' ms.')








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
        logs = log.CommonLog(logger, "rec_app/rec_log/offline_tarin.log")
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

        predict_matrix = self.predict(self.U, self.V) #??????????????

        end_offline_train = time.time()
        duration = str(round((end_offline_train - start_offline_train), 2))
        logs.info('Finish offline training. Total time : ' + str(duration) + ' s.\n' +
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------\n'+
                                                                             '---------------------------------------------------------------------------------------------------------------')

        start_offline_insert = time.time()
        print(predict_matrix)
        # ???
        # import pdb; pdb.set_trace()

        # 求item的top-K的索引
        # y = np.argmax(predict_matrix, axis=1)
        # print(y)
        # print(int(y[0]))
        # print(self.hash_item[int(y[0])])
        # import pdb; pdb.set_trace()
        topK_matrix = partition_arg_topK(predict_matrix, 10, axis=1)
        # print(topK_matrix)
        topK_matrix1 = np.array(topK_matrix)
        # for x1 in topK_matrix1:
        #     for y1 in x1:
        #         print(y1)
        #         y2 = self.hash_item[y1]
        #         topK_matrix1.itemset(y1, 1)
        #         print(y2)
        id = 0
        select_lastsql = 'SELECT * FROM rec_his ORDER BY create_time DESC limit 1'
        select_last = select_db(select_lastsql)
        id2 = select_last[0]['id'] + 1
        for x1 in range(len(topK_matrix1)):
            user_id = int(self.hash_user[x1])
            select_sql = 'SELECT * FROM rec WHERE user_id="%d"' % user_id  # %s（）或者用format
            select_result = select_db(select_sql)
            item_ids = ''
            for y1 in range(len(topK_matrix1[x1])):
                index2 =id
                m = topK_matrix1[x1][y1]
                # topK_matrix1[x1][y1] = self.hash_item[topK_matrix1[x1][y1]]
                item_id = int(self.hash_item[m])
                item_ids = item_ids + str(item_id) + ','
                # # insert_sql1 = "INSERT INTO rec(index2, user_id, item_ids, create_time) VALUES(%d, %d, %d, str_to_date(%s,'%%Y-%%m-%%d %%H:%%M:%%S'))" % (id, user_id, item_ids, now_time)

                # insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids) VALUES(%d, %d, %d)" % (id, user_id, item_ids)
                # # insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids, create_time) VALUES(%d, %d, %d, str_to_date(%s,'%%Y-%%m-%%d %%H:%%M:%%S'))" % (id, user_id, item_ids, now_time)
                # insert_db(insert_sql2)

            item_ids = item_ids[:-1]
            print(user_id, item_ids)
            if (select_result == ()):
                # insert
                insert_sql1 = "INSERT INTO rec(index2, user_id, item_ids) VALUES(%d, %d, '%s')" % (id, user_id, item_ids)
                insert_db(insert_sql1)
            else:
                # update
                update_sql = 'update rec set item_ids = "%s" where index2 = "%d"' % (item_ids, id)
                update_db(update_sql)

            insert_sql2 = "INSERT INTO rec_his(id, user_id, item_ids) VALUES(%d, %d, '%s')" % (id2, user_id, item_ids)
            insert_db(insert_sql2)

            id += 1
            id2 += 1


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




        self.predict_ = predict_matrix.getA().reshape(-1)  # .getA()将自身矩阵变量matrix转化为nd array类型的变量 reshape(-1)#改成一串，没有行列的列表
        # import pdb; pdb.set_trace()
        print("predict_new", self.predict_)
        self.predict_ = self.pre_handel(self.predict_, self.item_count)
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
