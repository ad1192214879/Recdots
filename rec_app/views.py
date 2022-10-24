from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
import pymysql
import sys
sys.path.append("/rec/rec_app")
sys.path.append("D:\\Recdots\\Recdots-master\\rec\\rec_app\\offline_online")

# rec_app.models import Rec, Item
#from id import id
from rec_app.offline_online import BPR
# data = Rec1.objects.filter(user_id=4124070)  #
# data = list(data)
# print(data)

from collections import defaultdict
import numpy as np
import time

import pymysql

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


# Create your views here.

# 用户数
user_count = 453
# 物品数
item_count = 23686
# k个主题，k数
latent_factors = 20

# hash映射
hash_user = dict()  # hash映射 0開始的id：本身id
user_hash = dict()  # hash映射 本身id：0開始的id
hash_item = dict()  # hash映射 0開始的id：本身id
item_hash = dict()  # hash映射 本身id：0開始的id
user_ratings = defaultdict(set)  # 自定义字典的value为集合（set）  例[('blue', {2, 4}), ('red', {1, 3})]
uid_clicked = defaultdict(set)   # 自定义字典的value为集合（set） [(uid,{clicked_iid1,clicked_iid2...}),(uid,{clicked_iid1,clicked_iid2...})]
# 随机设定的U，V矩阵(即公式中的Wuk和Hik)矩阵     Xui = W*HT
U = np.random.rand(user_count,
                   latent_factors) * 0.01  # 大小无所谓  np.random.rand返回一组user_count行 latent_factors列的（0,1]的随机数列表
V = np.random.rand(item_count, latent_factors) * 0.01
biasV = np.random.rand(item_count) * 0.01


select_sql = 'SELECT * FROM rec_his WHERE clicked_ids IS NOT NULL'
select_result = select_db(select_sql)
for i in range(len(select_result)):
    uid = select_result[i]['user_id']
    itemid = select_result[i]['clicked_ids']
    uid_clicked[int(uid)].add(int(itemid))

print(uid_clicked)


global bpr
bpr = BPR(hash_user, user_hash, hash_item, item_hash, user_ratings, U, V, biasV, uid_clicked)
bpr.main()


global test1
test1 = 1
# user = 4128772
# item = 73633
# bpr.online_train(user, item)


def test(request):
    if request.method == "GET":
        # return render(request, "config.html")
        print(request)
        print(request.GET)
        print(request.GET.get('uid'))
        # print(request.GET['uid'])
        print(123456789)
        uid = int(request.GET.get('uid'))
        select_sql = 'SELECT item_ids FROM rec WHERE user_id = "%d"'%uid
        select_result = select_db(select_sql)
        item_ids = select_result[0]['item_ids']
        if 'uid' in request.GET:
            return HttpResponse("连接成功"+item_ids)
        else:
            return HttpResponse("连接失败")
    else:
        data = request.POST
        print(data)
        # return HttpResponse("配置成功")
        return HttpResponse(test1)


def rec(request):
    # data = request.POST.get()
    # print(data)
    # print(111)
    if request.method == "GET":
        itemid = request.GET
        print(itemid)
        print(123)
        return render(request, "config.html")
    else:
        """获取推荐配置表单数据"""
        data = request.POST
        data = dict(data)
        print(data)
        # for item in meta['evaluation']:
        #     print(item)

        return HttpResponse("配置成功")
        # return redirect('http://127.0.0.1:8000/recResult/rec_result.html')
# else :
#     return redirect('')

def result(request):

    # logger = logging.getLogger()
    # logs = log.CommonLog(logger, "rec_app/reclog/log.log")
    global test
    test = 5555555555555555
    print(test)

    if request.is_ajax():
        itemid = request.POST.get('loadNewTable')
        uid = request.POST.get('uid')
        print("ID:", itemid, "uid:", uid)
        uid_clicked[int(uid)].add(int(itemid))
        A = time.time()
        bpr.online_train(int(uid), int(itemid))
        B = time.time()
        print((B - A) * 1000)
        # logs.info('id为'+uid+'的用户点击了'+'信息'+itemid)
        Data = {}
        return JsonResponse(Data)

    elif request.method == "GET":

        # logs.info('进入推荐结果页面')
        print(123456789)
        return render(request, "rec_result.html")

    elif request.method == "POST":
                uid = request.POST.get('uid')
                # print(uid)
                select_sql = 'SELECT * FROM rec WHERE user_id="%d"' % int(uid)  # %s（）或者用format
                select_result = select_db(select_sql)
                print(select_result)
                print(select_result[0]['user_id'])
                print(select_result[0]['item_ids'])
                # data = list(Rec.objects.filter(user_id = uid).all())  #
                # print(data)
                List = []
                items_list = select_result[0]['item_ids'].split(',')
                for i in range(10):
                    one = {'item_id' : '', 'item_name' : ''}
                    # n_id = data[i].item_ids
                    # data1 = list(Item.objects.filter(item_id=n_id).all())
                    # n_name = data1[0].item_name
                    n_id = items_list[i]
                    select_sql = 'SELECT * FROM item WHERE item_id="%d"' % int(n_id)  # %s（）或者用format
                    select_result1 = select_db(select_sql)
                    print(select_result1)
                    one['item_id'] = str(n_id)
                    one['item_name'] = select_result1[0]['item_name']
                    # print(one)
                    List.append(one)
                print(List)
                # List = List
                #
                # List = [{'item_id': '123', 'item_name': 'zhongguo918'},{'item_id': '1234', 'item_name': 'zhongguo918'},{'item_id': '12345', 'item_name': 'zhongguo918'},{'item_id': '123456', 'item_name': 'zhongguo918'}]

                return render(request, "rec_result.html", {'List':List, 'uid': uid})



def addToBehaviour(request):
    global uid
    global List
    if request.method == "POST":
        itemid = request.POST['itemid']
        # 在这里将itemid加到那个behaviour表里
        print(itemid)

        return redirect('../recResult')  #, args = {'List':List}



