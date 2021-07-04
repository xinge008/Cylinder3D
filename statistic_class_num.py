import numpy as np
import os
import tqdm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import shutil

def plot():
    class_map = {5: 'bus', 4: 'truck', 8: 'motorcyclist', 6: 'person', 7: 'bicyclist', 1: 'car', 0: 'unlabeled',
                 19: 'traffic-sign',18: 'pole', 17: 'terrain', 16: 'trunk', 15: 'vegetation', 9: 'road', 14: 'fence', 13: 'building',
                 12: 'other-ground', 11: 'sidewalk', 10: 'parking', 3: 'motorcycle', 2: 'bicycle'}
    file_num = 11
    for j in range(file_num):
        X = [str(i) for i in range(20)]
        Y = []
        fig = plt.figure()
        with open('out' + str(j).zfill(2) + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                Y.append(int(line))
        plt.bar(X, Y, 0.4, color="blue")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title('Sequence' + str(j).zfill(2))

        # plt.show()
        plt.savefig('barChart' + str(j).zfill(2) + '.jpg')


def statistc_class():
    learning_map = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10,
                    48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1,
                    253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}

    statistc_class_dic = {}
    file_num = 11
    root_path = '/Users/yexinyi/Desktop/VE450.nosync/data/merge/dataset/sequences'
    for i in range(20):
        statistc_class_dic[i] = 0

    for j in range(file_num):
        folder_path = os.path.join(root_path, str(j).zfill(2), 'labels')
        # folder_path = '/Users/yexinyi/Desktop/VE450.nosync/data/val_sub/dataset/sequences/08/labels'
        for root, dirs, files in os.walk(folder_path):
            for f in tqdm.tqdm(files):
                # filepath='/Users/yexinyi/Desktop/VE450.nosync/data/merge/dataset/sequences/00/labels/003907.label'
                filepath = os.path.join(root, f)
                annotated_data = np.fromfile(filepath, dtype=np.uint32).reshape((-1, 1))
                annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
                tmp = annotated_data.astype(np.uint8)
                for i in range(len(tmp)):
                    if tmp[i][0] in learning_map.keys():
                        if learning_map[tmp[i][0]] in statistc_class_dic.keys():
                            statistc_class_dic[learning_map[tmp[i][0]]] += 1

        with open('out_val' + str(j).zfill(2) + '.txt', 'w') as file:
            for i in statistc_class_dic.values():
                file.write(str(i) + '\n')

        break

def split_dataset():
    folder_path = '/Users/yexinyi/Desktop/VE450.nosync/data/merge/dataset/sequences/08/labels'
    folder_path2 = '/Users/yexinyi/Desktop/VE450.nosync/data/merge/dataset/sequences/08/velodyne'

    file_list = random.sample(os.listdir(folder_path), 814)  #0.2*4070
    for i in tqdm.tqdm(file_list):
        source = os.path.join(folder_path, i)
        target = os.path.join('/Users/yexinyi/Desktop/VE450.nosync/data/val_sub/dataset/sequences/08/labels', i)
        source2 = os.path.join(folder_path2, i[:-5]+'bin')
        target2 = os.path.join('/Users/yexinyi/Desktop/VE450.nosync/data/val_sub/dataset/sequences/08/velodyne', i[:-5]+'bin')
        shutil.copy(source, target)
        shutil.copy(source2, target2)



if __name__ == '__main__':
    statistc_class()
    # plot()
    # split_dataset()
