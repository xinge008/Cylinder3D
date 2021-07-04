# -*- coding:utf-8 -*-
# author: Xinge
# @file: dataset_nuscenes.py

import numpy as np
import torch
import numba as nb
from torch.utils import data
from dataloader.dataset_semantickitti import register_dataset


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


@register_dataset
class cylinder_dataset_nuscenes(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=0, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5],
                 scale_aug=False, transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz


# SemKITTI_label_name = {0: 'noise',
#                        1: 'animal',
#                        2: 'human.pedestrian.adult',
#                        3: 'human.pedestrian.child',
#                        4: 'human.pedestrian.construction_worker',
#                        5: 'human.pedestrian.personal_mobility',
#                        6: 'human.pedestrian.police_officer',
#                        7: 'human.pedestrian.stroller',
#                        8: 'human.pedestrian.wheelchair',
#                        9: 'movable_object.barrier',
#                        10: 'movable_object.debris',
#                        11: 'movable_object.pushable_pullable',
#                        12: 'movable_object.trafficcone',
#                        13: 'static_object.bicycle_rack',
#                        14: 'vehicle.bicycle',
#                        15: 'vehicle.bus.bendy',
#                        16: 'vehicle.bus.rigid',
#                        17: 'vehicle.car',
#                        18: 'vehicle.construction',
#                        19: 'vehicle.emergency.ambulance',
#                        20: 'vehicle.emergency.police',
#                        21: 'vehicle.motorcycle',
#                        22: 'vehicle.trailer',
#                        23: 'vehicle.truck',
#                        24: 'flat.driveable_surface',
#                        25: 'flat.other',
#                        26: 'flat.sidewalk',
#                        27: 'flat.terrain',
#                        28: 'static.manmade',
#                        29: 'static.other',
#                        30: 'static.vegetation',
#                        31: 'vehicle.ego'
#                        }
#
# SemKITTI_label_name_16 = {
#     0: 'noise',
#     1: 'barrier',
#     2: 'bicycle',
#     3: 'bus',
#     4: 'car',
#     5: 'construction_vehicle',
#     6: 'motorcycle',
#     7: 'pedestrian',
#     8: 'traffic_cone',
#     9: 'trailer',
#     10: 'truck',
#     11: 'driveable_surface',
#     12: 'other_flat',
#     13: 'sidewalk',
#     14: 'terrain',
#     15: 'manmade',
#     16: 'vegetation',
# }
#
# labels_mapping = {
#     1: 0,
#     5: 0,
#     7: 0,
#     8: 0,
#     10: 0,
#     11: 0,
#     13: 0,
#     19: 0,
#     20: 0,
#     0: 0,
#     29: 0,
#     31: 0,
#     9: 1,
#     14: 2,
#     15: 3,
#     16: 3,
#     17: 4,
#     18: 5,
#     21: 6,
#     2: 7,
#     3: 7,
#     4: 7,
#     6: 7,
#     12: 8,
#     22: 9,
#     23: 10,
#     24: 11,
#     25: 12,
#     26: 13,
#     27: 14,
#     28: 15,
#     30: 16
# }
