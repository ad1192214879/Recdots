from collections import Counter
import csv
from itertools import islice


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
    most_counterNum = collection_items.most_common(20)
    print(most_counterNum)
    print(most_counterNum[0])  #元组(id，count)
    print(most_counterNum[0][0])  #id
    print(type(most_counterNum))

    hot_set = set()
    for i in range(20):
        hot_set.add(most_counterNum[i][0])

    print(hot_set)
    return hot_set


if __name__ == '__main__':
    hot_set = hot_rec()
    print(hot_set)