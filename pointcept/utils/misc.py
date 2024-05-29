"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module

from sklearn import neighbors
from scipy.sparse.csgraph import shortest_path
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CategoryAverageMeter(object):
    """Computes and stores the average and current value for each category"""

    def __init__(self):
        self.categories = {}

    def reset(self):
        self.categories = {}

    def update(self, val, category, n=1):
        if category not in self.categories:
            self.categories[category] = {
                'val': 0,
                'sum': 0,
                'count': 0,
                'avg': 0,
            }
        self.categories[category]['val'] = val
        self.categories[category]['sum'] += val * n
        self.categories[category]['count'] += n
        self.categories[category]['avg'] = self.categories[category]['sum'] / self.categories[category]['count']

    def get_stats(self, category):
        if category in self.categories:
            return self.categories[category]
        else:
            return None


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass


def fp_fn_geodist(output, target, coord, dist_thresh=0.1, iou_thresh=0.1):
    # Compute geodesic distances
    geo_dists = gen_geo_dists(normalize_pc(coord))
    # GT keypoints
    gt_kps = np.where(target)[0]
    # Predicted keypoints
    pred_kps = np.where(output > iou_thresh)[0]
    # Compute fp and fn based on dist_thresh
    fp = np.count_nonzero(np.all(geo_dists[pred_kps][:, gt_kps] > dist_thresh, axis=-1))
    fn = np.count_nonzero(np.all(geo_dists[gt_kps][:, pred_kps] > dist_thresh, axis=-1))

    return fp, fn, len(gt_kps)


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc


def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False)
    return shortest_path(graph, directed=False)


def iou_per_category(data):
    category_stats = defaultdict(lambda: {'fp': 0, 'fn': 0, 'num_keypoints': 0})

    for entry in data.values():
        category = entry['category']
        category_stats[category]['fp'] += entry['fp']
        category_stats[category]['fn'] += entry['fn']
        category_stats[category]['num_keypoints'] += entry['num_keypoints']

    category_iou = {}
    for category, stats in category_stats.items():
        num_keypoints = stats['num_keypoints']
        fp = stats['fp']
        fn = stats['fn']
        iou = (num_keypoints - fn) / np.maximum(num_keypoints + fp, np.finfo(np.float64).eps)
        category_iou[category] = iou

    return category_iou
