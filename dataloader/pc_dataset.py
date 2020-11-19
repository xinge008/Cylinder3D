# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py 

import os
import numpy as np
from torch.utils import data
import yaml
import pickle

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]

@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml"):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple




@register_dataset
class SemKITTI_nusc(data.Dataset):

    SemKITTI_label_name = {
        0: 'noise',
         1: 'animal',
         2: 'human.pedestrian.adult',
         3: 'human.pedestrian.child',
         4: 'human.pedestrian.construction_worker',
         5: 'human.pedestrian.personal_mobility',
         6: 'human.pedestrian.police_officer',
         7: 'human.pedestrian.stroller',
         8: 'human.pedestrian.wheelchair',
         9: 'movable_object.barrier',
         10: 'movable_object.debris',
         11: 'movable_object.pushable_pullable',
         12: 'movable_object.trafficcone',
         13: 'static_object.bicycle_rack',
         14: 'vehicle.bicycle',
         15: 'vehicle.bus.bendy',
         16: 'vehicle.bus.rigid',
         17: 'vehicle.car',
         18: 'vehicle.construction',
         19: 'vehicle.emergency.ambulance',
         20: 'vehicle.emergency.police',
         21: 'vehicle.motorcycle',
         22: 'vehicle.trailer',
         23: 'vehicle.truck',
         24: 'flat.driveable_surface',
         25: 'flat.other',
         26: 'flat.sidewalk',
         27: 'flat.terrain',
         28: 'static.manmade',
         29: 'static.other',
         30: 'static.vegetation',
         31: 'vehicle.ego'
    }

    labels_mapping = {
        1:0,
        5:0,
        7:0,
        8:0,
        10:0,
        11:0,
        13:0,
        19:0,
        20:0,
        0:0,
        29:0,
        31:0,
        9:1,
        14:2,
        15:3,
        16:3,
        17:4,
        18:5,
        21:6,
        2:7,
        3:7,
        4:7,
        6:7,
        12:8,
        22:9,
        23:10,
        24:11,
        25:12,
        26:13,
        27:14,
        28:15,
        30:16
    }

    def __init__(self, data_path, nusc = None, imageset = 'train', info_path=None,
                 return_ref = False, splits = 'v1.0-trainval'):
        self.return_ref = return_ref

        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        self.imageset = imageset
        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):

        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1,1])
        points_label = np.vectorize(self.labels_mapping.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:,:3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:,3],)
        return data_tuple



def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label


# load Semantic KITTI class info

def get_SemKITTI_label_name(label_mapping):

    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name