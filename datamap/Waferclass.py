
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import os

class WaferDataset(Dataset):
    def __init__(self, df, wafer_list, label_col='IsScratchDie', is_test=False, cache_dir=None):
        self.df = df
        self.wafer_list = wafer_list
        self.label_col = label_col
        self.is_test = is_test
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join('datamap', 'data', 'cache')

        os.makedirs(os.path.join(self.cache_dir, "images"), exist_ok=True)
        if not is_test:
            os.makedirs(os.path.join(self.cache_dir, "labels"), exist_ok=True)

        self.image_cache = {}
        self.label_cache = {}

        for name in wafer_list:
            img_path = os.path.join(self.cache_dir, "images", f"{name}.pt")
            lbl_path = os.path.join(self.cache_dir, "labels", f"{name}.pt")

            if os.path.exists(img_path):
                image = torch.load(img_path)
            else:
                image = self.__create_wafer_map(df, name, 'IsGoodDie')
                image = torch.tensor(image).unsqueeze(0).float()
                image = TF.resize(image, size=(256, 256), interpolation=TF.InterpolationMode.BILINEAR)
                torch.save(image, img_path)

            self.image_cache[name] = image

            if not is_test:
                if os.path.exists(lbl_path):
                    label = torch.load(lbl_path)
                else:
                    label = self.__create_wafer_map(df, name, self.label_col)
                    label = torch.tensor(label).long()
                    label = TF.resize(label.unsqueeze(0), size=(256, 256), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
                    torch.save(label, lbl_path)

                self.label_cache[name] = label

    def __len__(self):
        return len(self.wafer_list)

    def __create_wafer_map(self,df, wafer_name, col, fill=0):
        wafer_df = df[df['WaferName'] == wafer_name]
        min_x, max_x = wafer_df['DieX'].min(), wafer_df['DieX'].max()
        min_y, max_y = wafer_df['DieY'].min(), wafer_df['DieY'].max()
        width, height = max_x - min_x + 1, max_y - min_y + 1
        wafer_map = np.full((height, width), fill, dtype=np.float32)

        for _, row in wafer_df.iterrows():
            x, y = int(row['DieX'] - min_x), int(row['DieY'] - min_y)
            wafer_map[y, x] = row[col]
        return wafer_map

    def __getitem__(self, idx):
        name = self.wafer_list[idx]
        image = self.image_cache[name]
        if not self.is_test:
            label = self.label_cache[name]
        else:
            label = torch.zeros((256, 256), dtype=torch.long)
        return image, label