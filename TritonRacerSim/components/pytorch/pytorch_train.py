import torch
import json
from os import path
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim import SGD, Adam
import torch.nn as nn
from skimage import io
import numpy as np

from TritonRacerSim.components.pytorch.pytorch_datasets import prepare_data, train_val_test_split
from TritonRacerSim.components.pytorch.pytorch_models import get_baseline_cnn

''' Training on a baseline CNN that takes a image and predicts some regression outputs '''
class Trainer:
    def __init__(self, data_paths: list, hyperparams: dict) -> None:
        trans = transforms.ToTensor()
        data = prepare_data(data_paths, trans)
        datasets = train_val_test_split(data)
        self.batch_size = hyperparams['batch_size']
        self.early_stop_patience = hyperparams['early_stop_patience']
        self.max_epoch = hyperparams['max_epoch']
        self.train_loader, self.val_loader, self.test_loader = (DataLoader(dataset, self.batch_size,
                                                shuffle=hyperparams['shuffle'], 
                                                num_workers=hyperparams['num_workers']) 
                                                for dataset in datasets)
        self.model = get_baseline_cnn()
        self.optm = self.__get_optimizer__(hyperparams, self.model)
        self.device = self.__get_device__()
        self.criterion = self.__get_loss__(hyperparams)
        self.val_losses = []
        self.train_losses = []
        self._last_best_loss = float('inf')
        self.use_tensorrt = hyperparams['use_tensorrt']
    
    def __get_optimizer__(hpps: dict, model):
        lr = hpps.get('learning_rate')
        wd = hpps.get('weight_decay')
        mom = hpps.get('momentum')
        optm_name = hpps.get('optimizer').strip()
        if optm_name == 'sgd':
            return SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        elif optm_name == 'adam':
            return Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise Exception(f'Unknown optimizer type "{optm_name}" in hyperparam.')

    def __get_loss__(self, hpps: dict):
        return nn.MSELoss()

    def __get_device__(self):
        if torch.cuda.is_available():
            print('Using CUDA.')
            return torch.device('cuda')
        print('CUDA not found. Using CPU for training.')       
        return torch.device('cpu')

    ''' Overload this method to train on other labels '''
    def __get_labels__(self, batched_data:dict):
        return batched_data['mux/steering'], batched_data['gym/speed']

    def train(self):
        self.model = self.model.to(self.device)
        for e in range(self.max_epoch):
            self.model.train()

            ''' Train '''
            running_loss = 0.0
            for idx_batch, data in enumerate(self.train_loader):
                x = data['image'].to(self.device, dtype=torch.float32)
                self.image_size = x.size
                y = self.__get_labels__(data).to(self.device, dtype=torch.float32)

                y_preds = self.model(x)
                loss = self.criterion(y_preds, y)

                self.optm.zero_grad()
                loss.backward()
                self.optm.step()

                loss_str = "{:.4f}".format(loss.item())
                print(f"\r[Epoch {e+1}, Training {idx_batch}/{len(self.train_loader)}]: training loss (current batch): {loss_str}", end='')
                running_loss += loss.item()
            train_loss = running_loss / len(self.train_loader)

            ''' Validate '''
            running_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for idx_batch, data in enumerate(self.val_loader):
                    x = data['image'].to(self.device, dtype=torch.float32)
                    y = self.__get_labels__(data).to(self.device, dtype=torch.float32)

                    y_preds = self.model(x)
                    loss = self.criterion(y_preds, y)

                    loss_str = "{:.4f}".format(loss.item())
                    print(f"\r[Epoch {e+1}, Validating {idx_batch}/{len(self.train_loader)}]: validation loss (current batch): {loss_str}", end='')
                    running_loss += loss.item()             
            val_loss = running_loss / len(self.val_loader)

            train_loss_str = "{:.4f}".format(train_loss)
            val_loss_str = "{:.4f}".format(val_loss)
            print(f"[Epoch {e+1}]: validation loss: {train_loss_str}, training loss: {val_loss_str}")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            early_stop = self.__on_epoch_end__(self.model, self.model_path)
            if early_stop:
                print("[Early Stop]")
                return

    def __check_if_early_stop__(self):
        if len(self.val_losses) < self.early_stop_patience: return False

        last_losses = self.val_losses[-self.early_stop_patience:]
        for loss in last_losses:
            if loss < self._last_best_loss:
                self._last_best_loss = loss
                return False
        
        return True

    def __check_if_save_model__(self):
        if len(self.val_losses) < 2: 
            return True
        return self.val_losses[-1] < self.val_losses[-2]

    ''' Overload this method to customize end-of-epoch behavior '''
    def __on_epoch_end__(self, model, model_path):
        if self.__check_if_save_model__():
            print(f'Validation loss decreased. Saving model to {model_path}.')
            if self.use_tensorrt:
                from torch2trt import torch2trt
                dummy_x = torch.ones(self.image_size).to(self.device)
                model_trt = torch2trt(model, [dummy_x])
                torch.save(model_trt.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        else:
            print(f'Validation loss did not decrease.')
        return self.__check_if_early_stop__()