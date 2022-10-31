from collections import Counter
import csv
from itertools import islice


def hot_rec():

    import pandas as pd
    filename = 'D:\\Recdots\\Recdots-master\\rec\\rec_app\\user_item.csv'
    total_lines = sum(1 for line in open(filename, encoding='utf-8'))
    print('The total lines is ', total_lines)

    hot_list = [[0] * 50 for _ in range(7)]

    csv_result = pd.read_csv('D:\\Recdots\\Recdots-master\\rec\\rec_app\\user_item.csv')

    if total_lines >= 1000000:
        start_line = total_lines - 1000000
        end_line = total_lines
    else:
        start_line = 0
        end_line = total_lines

    csv_result1 = csv_result.iloc[start_line:end_line, 0:4]
    row_list = csv_result1.values.tolist()

    for i in range(1, 8):
        item_list = []
        for m in range(end_line - start_line):
            try:
                item = int(row_list[m][1])
                type = int(row_list[m][3])
                if type == i:
                    item_list.append(item)
            except:
                print("error")
        collection_items = Counter(item_list)
        # 还可以输出频率最大的n个元素,类型为list
        most_counterNum = collection_items.most_common(50)
        for j in range(50):
            hot_list[i - 1][j] = str(most_counterNum[j][0]) + '/' + str(i)
    print(hot_list)
    return hot_list


if __name__ == '__main__':
    hot_list = hot_rec()
    print(hot_list)