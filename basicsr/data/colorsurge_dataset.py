import cv2
import random
import time
import numpy as np
import torch
from torch.utils import data as data
import ast  
import pandas as pd
import imageio

from basicsr.data.transforms import rgb2lab
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.fmix import sample_mask

@DATASET_REGISTRY.register()
class ColorSurge_Dataset(data.Dataset):
    """
    Dataset used for Lab colorizaion
    """

    def __init__(self, meta_info_file=[], frame_len = 16):
        super(ColorSurge_Dataset, self).__init__()

        self.frame_len = frame_len
        self.meta_info_file = meta_info_file
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = {'type': 'disk'}
        assert meta_info_file is not None
        if not isinstance(meta_info_file, list):
            meta_info_file = [meta_info_file]
        self.paths = []
        for meta_info in meta_info_file:
            try:
                with open(meta_info, 'r') as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            try:
                                path_list = ast.literal_eval(line)
                                if isinstance(path_list, list):
                                    self.paths.append(path_list)
                            except (SyntaxError, ValueError):
                                print(f"Warning: Unable to parse line: {line}")

            except Exception as e:
                print(f"Error reading {meta_info_file}: {e}")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        gt_path = gt_path[:self.frame_len]
        # avoid errors caused by high latency in reading files
        retry = 3
        frame_gt_list = []
        
        while retry > 0:
            try:
                for i in range(self.frame_len):
                    img_bytes= self.file_client.get(gt_path[i], 'gt')
                    frame_gt_list.append(imfrombytes(img_bytes, float32=True))
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        return_d = {
            'gt_path': gt_path
        }
        return return_d


    def __len__(self):
        return len(self.paths)

