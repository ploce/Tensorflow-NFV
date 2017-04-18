#coding=utf-8
import copy
import numpy as np

NODE_NUM = 24

def write_line(f, line):
    for i in range(0, len(line) - 1):
        f.write(str(line[i]))
        f.write(' ')
    f.write(str(line[len(line) - 1]))
    f.write('\n')


def change_topo_state(topo, src_id, des_id):
    '''把src_id与des_id之间的边从topo图中删除'''
    topo[src_id * NODE_NUM + des_id] = 0
    topo[des_id * NODE_NUM + src_id] = 0
    return topo


def process_train_data():
    '''对训练数据做预处理'''

    #处理topo图
    topo = np.array([[0]*NODE_NUM for i in range(NODE_NUM)])

    f_topo = open('./train_data/LinkInfo.txt', 'r')
    train_data_x = open('./train_data_x.txt', 'w')
    train_data_y = open('./train_data_y.txt', 'w')

    linkinfo = f_topo.readlines()
    for i in range(2, len(linkinfo)):
        line = linkinfo[i].split(" ")
        line = np.array(line)
        line = line.astype(int)
        x = line[1]
        y = line[2]
        topo[x][y] = 1
        topo[y][x] = 1
    topo = topo.reshape(NODE_NUM * NODE_NUM)

    #处理路径信息，在图后面添加当前ID，目的ID，链长
    for i in range(5, 15):
        name = './train_data/' + str(i) + 'tensorflowLayerResult.txt'
        f = open(name, 'r')
        content = f.readlines()
        for j in range(2, len(content) - 1):
            print('Link Len : %d, n : %d' % (i, j))
            line = content[j].split()
            line = np.array(line)
            line = line.astype(int)
            topo_temp = copy.deepcopy(topo)
            n = i - 2
            for k in range(2, len(line) - 1):
                info = np.array([line[k], line[1], n])
                data_x = np.concatenate((topo_temp, info))
                data_y = np.array([0] * NODE_NUM)
                data_y[line[k + 1]] = 1

                write_line(train_data_x, data_x)
                write_line(train_data_y, data_y)

                topo_temp = change_topo_state(topo_temp, line[k], line[k + 1])
                if n > 0:
                    n = n - 1
    
    f_topo.close()
    train_data_x.close()
    train_data_y.close()


if __name__ == "__main__":
    process_train_data()
