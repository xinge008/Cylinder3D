import numpy as np
import os
import tqdm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import shutil

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

# quadratic
def quadratic_func(x):
    return x ** 0.5

# triple
def triple_func(x):
    return x ** (1/3)

# ln
def ln_func(x):
    return np.log(x)

def cal_cell_num(func_type):
    file_num = 11
    root_path = '/Users/yexinyi/Desktop/VE450.nosync/data/merge/dataset/sequences'

    max_bound = np.asarray([50, 3.1415926, 2])
    min_bound = np.asarray([0, -3.1415926, -4])
    crop_range = max_bound - min_bound
    cur_grid_size = np.asarray([480, 360, 32])

    # for plot
    dim1 = cur_grid_size[0]
    dim2 = cur_grid_size[1]
    dim3 = cur_grid_size[2]

    total_cell_list = [0, 0, 0, 0, 0]
    full_cell_list = [0, 0, 0, 0, 0]
    res = []
    total_file_num = 0

    for j in range(file_num):
        if j == 8:
            continue
        folder_path = os.path.join(root_path, str(j).zfill(2), 'velodyne')
        # folder_path = '/Users/yexinyi/Desktop/VE450.nosync/data/merge/dataset/sequences/01/velodyne'
        for root, dirs, files in os.walk(folder_path):
            for f in tqdm.tqdm(files):
                # filepath='/Users/yexinyi/Desktop/VE450.nosync/data/merge/dataset/sequences/00/velodyne/003907.bin'
                filepath = os.path.join(root, f)
                raw_data = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))
                xyz_pol = cart2polar(raw_data[:, :3])

                total_file_num += 1
                if func_type == "quadratic":
                    x_clip = np.clip(xyz_pol, min_bound, max_bound)
                    a = (cur_grid_size[0] - 1) / (quadratic_func(max_bound[0] - min_bound[0]))
                    tmp_y = a * quadratic_func(x_clip[:, 0] - min_bound[0])
                    grid_ind1 = (np.floor(tmp_y)).astype(np.int)
                    grid_ind2 = (np.floor((x_clip[:, 1:] - min_bound[1:]) / ((max_bound[1:] - min_bound[1:]) / (
                                cur_grid_size[1:] - 1)))).astype(np.int)
                    grid_ind = np.concatenate((grid_ind1.reshape((grid_ind1.shape[0], 1)), grid_ind2), axis=1)

                    for i in range(5):
                        left_x = i * 10
                        right_x = (i + 1) * 10
                        left_y = np.floor(a * quadratic_func(left_x))
                        right_y = np.floor(a * quadratic_func(right_x))
                        unique_cell = np.unique(grid_ind, axis=0)
                        full_cell_num = np.sum(np.logical_and((unique_cell[:, 0] >= left_y), (unique_cell[:, 0] < right_y)))
                        full_cell_list[i] += full_cell_num

                elif func_type == "triple":
                    x_clip = np.clip(xyz_pol, min_bound, max_bound)
                    a = (cur_grid_size[0] - 1) / (triple_func(max_bound[0] - min_bound[0]))
                    tmp_y = a * triple_func(x_clip[:, 0] - min_bound[0])
                    grid_ind1 = (np.floor(tmp_y)).astype(np.int)
                    grid_ind2 = (np.floor((x_clip[:, 1:] - min_bound[1:]) / ((max_bound[1:] - min_bound[1:]) / (
                            cur_grid_size[1:] - 1)))).astype(np.int)
                    grid_ind = np.concatenate((grid_ind1.reshape((grid_ind1.shape[0], 1)), grid_ind2), axis=1)

                    for i in range(5):
                        left_x = i * 10
                        right_x = (i + 1) * 10
                        left_y = np.floor(a * triple_func(left_x))
                        right_y = np.floor(a * triple_func(right_x))
                        unique_cell = np.unique(grid_ind, axis=0)
                        full_cell_num = np.sum(np.logical_and((unique_cell[:, 0] >= left_y), (unique_cell[:, 0] < right_y)))
                        full_cell_list[i] += full_cell_num

                elif func_type == "ln":
                    x_clip = np.clip(xyz_pol, min_bound, max_bound)
                    a = (cur_grid_size[0] - 1) / (ln_func(max_bound[0] + 1 - min_bound[0]))
                    tmp_y = a * ln_func(x_clip[:, 0] + 1 - min_bound[0])
                    grid_ind1 = (np.floor(tmp_y)).astype(np.int)
                    grid_ind2 = (np.floor((x_clip[:, 1:] - min_bound[1:]) / ((max_bound[1:] - min_bound[1:]) / (
                            cur_grid_size[1:] - 1)))).astype(np.int)
                    grid_ind = np.concatenate((grid_ind1.reshape((grid_ind1.shape[0], 1)), grid_ind2), axis=1)
                    for i in range(5):
                        left_x = i * 10
                        right_x = (i + 1) * 10
                        left_y = np.floor(a * ln_func(left_x))
                        right_y = np.floor(a * ln_func(right_x))
                        unique_cell = np.unique(grid_ind, axis=0)
                        full_cell_num = np.sum(np.logical_and((unique_cell[:, 0] >= left_y), (unique_cell[:, 0] < right_y)))
                        full_cell_list[i] += full_cell_num

                elif func_type == "original":
                    intervals = crop_range / (cur_grid_size - 1)
                    if (intervals == 0).any(): print("Zero interval!")
                    x_clip = np.clip(xyz_pol, min_bound, max_bound)
                    grid_ind = (np.floor((x_clip - min_bound) / intervals)).astype(np.int)

                    for i in range(5):
                        left_x = i * 10
                        right_x = (i + 1) * 10
                        left_y = np.floor(left_x / intervals[0])
                        right_y = np.floor(right_x / intervals[0])
                        unique_cell = np.unique(grid_ind, axis=0)
                        full_cell_num = np.sum(
                            np.logical_and((unique_cell[:, 0] >= left_y), (unique_cell[:, 0] < right_y)))
                        full_cell_list[i] += full_cell_num


    for j in range(5):
        left_x = j * 10
        right_x = (j + 1) * 10
        if func_type == "quadratic":
            a = (cur_grid_size[0] - 1) / (quadratic_func(max_bound[0] - min_bound[0]))
            left_y = np.floor(a * quadratic_func(left_x))
            right_y = np.floor(a * quadratic_func(right_x))
            total_cell_list[j] = (right_y - left_y) * dim2 * dim3 * total_file_num

        elif func_type == "triple":
            a = (cur_grid_size[0] - 1) / (triple_func(max_bound[0] - min_bound[0]))
            left_y = np.floor(a * triple_func(left_x))
            right_y = np.floor(a * triple_func(right_x))
            total_cell_list[j] = (right_y - left_y) * dim2 * dim3 * total_file_num

        elif func_type == "ln":
            a = (cur_grid_size[0] - 1) / (ln_func(max_bound[0] + 1 - min_bound[0]))
            left_y = np.floor(a * ln_func(left_x + 1))
            right_y = np.floor(a * ln_func(right_x + 1))
            total_cell_list[j] = (right_y - left_y) * dim2 * dim3 * total_file_num

        elif func_type == "original":
            intervals = crop_range / (cur_grid_size - 1)
            left_y = np.floor((np.clip(left_x, min_bound[0], max_bound[0]) - min_bound[0]) / intervals[0])
            right_y = np.floor((np.clip(right_x, min_bound[0], max_bound[0]) - min_bound[0]) / intervals[0])
            total_cell_list[j] = (right_y - left_y) * dim2 * dim3 * total_file_num

        res.append(full_cell_list[j] / total_cell_list[j])
        print("res:", res)

        with open(func_type + '.txt', 'w') as file:
            for i in res:
                file.write(str(i) + '\n')

if __name__ == '__main__':
    func_type = ["quadratic", "triple", "ln", "original"]
    cal_cell_num(func_type[2])
