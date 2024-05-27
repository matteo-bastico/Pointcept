"""
KeyPointNet Dataset

get point clouds at https://github.com/qq456cvb/KeypointNet

Author: Matteo Bastico (matteo.bastico@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import pointops
import torch
import json
from torch.utils.data import Dataset
from copy import deepcopy


from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=float)
    colors = np.array(data[:, -1], dtype=int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    # Stack them
    return np.hstack((pc, colors)).astype(np.float32)


@DATASETS.register_module()
class KeyPointNetDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/keypointnet",
        category='all',
        class_id2names=None,
        transform=None,
        num_points=2048,
        uniform_sampling=True,
        save_record=True,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        print(test_cfg)
        super().__init__()
        self.data_root = data_root
        self.category = category
        self.class_id2name = class_id2names
        self.class_name2id = {v: k for k, v in class_id2names.items()}
        self.class_names = dict(zip(class_id2names.keys(), range(len(class_id2names))))
        self.split = split
        self.num_point = num_points
        self.uniform_sampling = uniform_sampling
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        self.keypoints = self.get_keypoints()

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )
        # check, prepare record
        record_name = f"keypointnet_{self.split}_{self.category}"
        if num_points is not None:
            record_name += f"_{num_points}points"
            if uniform_sampling:
                record_name += "_uniform"
        record_path = os.path.join(self.data_root, f"{record_name}.pth")
        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data = torch.load(record_path)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data = {}
            for idx in range(len(self.data_list)):
                data_name = self.data_list[idx]
                logger.info(f"Parsing data [{idx}/{len(self.data_list)}]: {data_name}")
                self.data[data_name] = self.get_data(idx)
            if save_record:
                torch.save(self.data, record_path)

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return self.data[data_name]
        else:
            # Separate category and name
            data_shape = data_name.split('-')[0]
            data_name = data_name.split('-')[-1].rstrip('\n')
            data_path = os.path.join(
                self.data_root, 'pcds', data_shape, data_name + ".pcd"
            )
            data = naive_read_pcd(data_path)
            if self.num_point is not None:
                if self.uniform_sampling:
                    with torch.no_grad():
                        mask = pointops.farthest_point_sampling(
                            torch.tensor(data).float().cuda(),
                            torch.tensor([len(data)]).long().cuda(),
                            torch.tensor([self.num_point]).long().cuda(),
                        )
                    data = data[mask.cpu()]
                else:
                    data = data[: self.num_point]
            coord, color = data[:, 0:3], data[:, 3:6]
            category = np.array([self.class_names[data_shape]])
            # Parse annotations
            keypoints = self.keypoints[data_name]
            segment = np.zeros((coord.shape[0],), dtype=np.int32)
            segment[keypoints] = 1
            return dict(coord=coord, color=color, category=category, segment=segment, name=data_name)

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(
            self.data_root, "keypointnet_{}.txt".format(self.split)
        )
        data_list = np.loadtxt(split_path, dtype="str")
        if self.category == 'all':
            return data_list
        else:
            data_list = [entry for entry in data_list if entry.startswith(self.class_name2id[self.category])]
            return data_list

    def get_keypoints(self):
        annots = json.load(open(os.path.join(self.data_root, 'annotations', self.category + '.json')))
        keypoints = dict(
            [(annot['model_id'], [kp_info['pcd_info']['point_index'] for kp_info in annot['keypoints']]) for annot in
             annots])
        return keypoints

    def __len__(self):
        return len(self.data_list) * self.loop

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)
    
    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    # From DefaultDataset, modified since we need also the category in the metrics computation
    def prepare_test_data(self, idx):
        # TODO: Why fragment? How do we prepare test data for Keypoints prediction
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(
            segment=data_dict.pop("segment"),
            name=data_dict.pop("name"),
            category=data_dict.pop("category")
        )
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part
        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

