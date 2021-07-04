# -*- coding:utf-8 -*-
# author: Ptzu
# @file: demo_folder.py

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")


def build_dataset(dataset_config,
                  data_dir,
                  grid_size=[480, 360, 32],
                  demo_label_dir=None):

    if demo_label_dir == '':
        imageset = "demo"
    else:
        imageset = "val"
    label_mapping = dataset_config["label_mapping"]

    SemKITTI_demo = get_pc_model_class('SemKITTI_demo')

    demo_pt_dataset = SemKITTI_demo(data_dir, imageset=imageset,
                              return_ref=True, label_mapping=label_mapping, demo_label_path=demo_label_dir)

    demo_dataset = get_model_class(dataset_config['dataset_type'])(
        demo_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )
    demo_dataset_loader = torch.utils.data.DataLoader(dataset=demo_dataset,
                                                     batch_size=1,
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=False,
                                                     num_workers=4)

    return demo_dataset_loader

def main(args):
    pytorch_device = torch.device('cuda:0')
    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    data_dir = args.demo_folder
    demo_label_dir = args.demo_label_folder
    save_dir = args.save_folder + "/"

    demo_batch_size = 1
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    model_load_path = train_hypers['model_load_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    demo_dataset_loader = build_dataset(dataset_config, data_dir, grid_size=grid_size, demo_label_dir=demo_label_dir)
    with open(dataset_config["label_mapping"], 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    inv_learning_map = semkittiyaml['learning_map_inv']

    my_model.eval()
    hist_list = []
    demo_loss_list = []
    with torch.no_grad():
        for i_iter_demo, (_, demo_vox_label, demo_grid, demo_pt_labs, demo_pt_fea) in enumerate(
                demo_dataset_loader):
            demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              demo_pt_fea]
            demo_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in demo_grid]
            demo_label_tensor = demo_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(demo_pt_fea_ten, demo_grid_ten, demo_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), demo_label_tensor,
                                  ignore=0) + loss_func(predict_labels.detach(), demo_label_tensor)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, i_demo_grid in enumerate(demo_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, demo_grid[count][:, 0], demo_grid[count][:, 1],
                                                    demo_grid[count][:, 2]], demo_pt_labs[count],
                                                unique_label))
                inv_labels = np.vectorize(inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]) 
                inv_labels = inv_labels.astype('uint32')
                outputPath = save_dir + str(i_iter_demo).zfill(6) + '.label'
                inv_labels.tofile(outputPath)
                print("save " + outputPath)
            demo_loss_list.append(loss.detach().cpu().numpy())

    if demo_label_dir != '':
        my_model.train()
        iou = per_class_iu(sum(hist_list))
        print('Validation per class iou: ')
        for class_name, class_iou in zip(unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100
        del demo_vox_label, demo_grid, demo_pt_fea, demo_grid_ten

        print('Current val miou is %.3f' %
              (val_miou))
        print('Current val loss is %.3f' %
              (np.mean(demo_loss_list)))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('--demo-folder', type=str, default='', help='path to the folder containing demo lidar scans', required=True)
    parser.add_argument('--save-folder', type=str, default='', help='path to save your result', required=True)
    parser.add_argument('--demo-label-folder', type=str, default='', help='path to the folder containing demo labels')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
