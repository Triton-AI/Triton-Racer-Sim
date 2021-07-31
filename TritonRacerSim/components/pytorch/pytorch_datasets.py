import torch
import json
from os import path
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
from skimage import io
import numpy as np


class Tub(Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform
        self.dir = dir
        # Iterate through the directory and generate a list of all records
        self.record_list = []
        print(f'Reading records from {dir}: ', end='')
        self.record_count = 1
        while True:
            json_name = self.__get_json_name(self.dir, self.record_count)
            img_name = self.__get_image_name(self.dir, self.record_count)
            if path.exists(json_name) and path.exists(img_name):
                self.record_list.append((img_name, json_name))
                self.record_count += 1
                if self.record_count % 100 == 0:
                    print(f'\rfound {self.record_count} records...', end='')
            else:
                self.record_count -= 1
                break
            print(f'\rfound {self.record_count} records.')


    def __get_json_name(self, dir, idx):
        return path.join(dir, f'record_{idx}.json')
    

    def __get_image_name(self, dir, idx):
        return path.join(dir, f'img_{idx}.jpg')


    def __len__(self):
        return self.record_count

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #       idx = idx.tolist()
        image_path, json_path = self.record_list[idx]
        image = io.imread(image_path)
        with open(json_path, 'r') as json_file:
            json_dic = json.load(json_file)

        if self.transform:
            image = self.transform(image)
        item = {'image': image}
        item.update(self.__get_examples_from_json__(json_dic))
        item.update(self.__get_targets_from_json__(json_dic))
        return item
        
    def __get_examples_from_json__(json_dic):
        to_get=['mux/throttle', 'mux/steering', 'mux/break', 
        'gym/speed','loc/segment', 'gym/x', 
        'gym/y', 'gym/z', 'gym/cte', 'loc/cte', 
        'loc/break_indicator']

        return {key: json_dic[key] for key in to_get}

    def __get_targets_from_json__(json_dic):
        pass




# Prepare dataset
def prepare_data(paths: list, dataset_type=Tub, transform=None) -> ConcatDataset:
    datasets=[]
    for path in paths:
        datasets.append(dataset_type(path), transform)
    return ConcatDataset(datasets)

def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    num_data = len(dataset)
    num_train = round(num_data * train_ratio)
    num_val = round(num_data * val_ratio)
    num_test = round(num_data * test_ratio)
    return random_split(dataset, [num_train, num_val, num_test])