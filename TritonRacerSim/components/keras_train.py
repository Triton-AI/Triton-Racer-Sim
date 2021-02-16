
from os import path
import numpy as np
from PIL import Image
import json
from enum import Enum
import time
import os
from abc import abstractmethod
from numpy.core import overrides

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, Concatenate, MaxPooling2D, LSTM, Embedding
from tensorflow.keras import optimizers, losses
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers.experimental.preprocessing import Resizing

from TritonRacerSim.components.controller import DriveMode
from TritonRacerSim.utils.types import ModelType
from TritonRacerSim.components.component import Component

class DataLoader:
    '''Load img and json records from record folder'''
    def __init__(self, *paths):
        self.paths = paths
        for data_path in paths:
            if not path.exists(data_path):
                raise FileNotFoundError(f'Folder does not exists: {data_path}')
        self.dataset = []
        self.train_dataset = None
        self.val_dataset = None

    def load(self, train_val_split = 0.8, batch_size = 128):
        print ('Loading records...')
        for data_path in self.paths:
            i = 1
            while True:
                try:
                    # Obtain img as array
                    img_path = path.join(data_path, self.get_img_name(i))
                    img_arr = np.asarray(Image.open(img_path),dtype=np.float32)
                    img_arr /= 255

                    # Obtain labels and feature vectors as arrays
                    record_path = path.join(data_path, self.get_record_name(i))
                    record={}
                    with open(record_path) as f:
                        record= json.load(f)
                    labels = np.asarray(self.get_labels_from_record(record),dtype=np.float32)
                    feature_vectors = np.asarray(self.get_features_from_record(record), dtype=np.float32)

                    self.dataset.append((img_arr, feature_vectors, labels))
                    # print (labels)
                    i += 1
                    if i % 100 == 0: print(f"\rLoading {i} records...",end="" )
                except FileNotFoundError:
                    # print (f'Loaded {i-1} records in {data_path}')
                    break

        print (f'Loaded {len(self.dataset)} records.')

    def shuffle_batch(self, batch_size):
        SHUFFLE_BUFFER_SIZE = 5000
        self.train_dataset_batch = self.train_dataset.unbatch().shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size, drop_remainder=True)
        self.val_dataset_batch = self.val_dataset.unbatch().shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size, drop_remainder=True)

    def split_train_val(self, split = 0.8, batch_size = 64):
        assert 0 < split <= 1
        from sklearn.model_selection import train_test_split
        train_set, val_set = train_test_split(self.dataset, train_size = split)

        train_examples = []
        train_example_vecs = []
        train_labels = []

        val_examples = []
        val_example_vecs = []
        val_labels = []
        
        for data in train_set:
            if data[1].size: # Has feature vectors?
                train_examples.append(data[0])
                train_example_vecs.append(data[1])
            else:
                train_examples.append(data[0])
            train_labels.append(data[2])

        for data in val_set:
            val_examples.append(data[0])
            if data[1].size:
                val_example_vecs.append(data[1])
            val_labels.append(data[2])

        train_examples = np.stack(train_examples, axis=0)
        train_labels = np.stack(train_labels, axis=0)
        val_examples = np.stack(val_examples, axis=0)
        val_labels = np.stack(val_labels, axis=0)

        if train_example_vecs:
            train_example_vecs = np.stack(train_example_vecs, axis=0)
            val_example_vecs = np.stack(val_example_vecs, axis=0)
            self.train_dataset = tf.data.Dataset.from_tensors(((train_examples, train_example_vecs), train_labels))
            self.val_dataset = tf.data.Dataset.from_tensors(((val_examples, val_example_vecs), val_labels)) 
        else:
            self.train_dataset = tf.data.Dataset.from_tensors((train_examples, train_labels))
            self.val_dataset = tf.data.Dataset.from_tensors((val_examples, val_labels))        

    def get_img_name(self, idx):
        return f'img_{idx}.jpg'

    def get_record_name(self, idx):
        return f'record_{idx}.json'

    def get_labels_from_record(self, record={}):
        return record['mux/steering'], record['mux/throttle'] # Adjust the input range to be [0, 1]

    def get_features_from_record(self,record={}):
        '''Any additional features are we looking for?'''
        # return record['gym/speed'], record['gym/cte']
        return None
    

class Keras_2D_CNN(Component):
    '''2D CNN models'''
    def __init__(self, input_shape, num_outputs, num_feature_vectors = 0):
        pass
    
    @staticmethod
    def get_model(input_shape, num_outputs, num_feature_vectors = 0):
        inputs = Input(shape=input_shape, name='img_input')
        
        drop = 0.1

        x = Rescaling(scale=1./255)(inputs)
        x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2),activation='relu', name='conv1')(inputs)
        x = Dropout(drop)(x)
        x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2),activation='relu', name='conv2')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv3')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv4')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv5')(x)
        x = Dropout(drop)(x)
        
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv6')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv7')(x)
        x = Dropout(drop)(x)
        x = Flatten(name='flatten1')(x)
        z = x
        if num_feature_vectors > 0:
            feature_inputs = Input(shape=(num_feature_vectors,), name='feature_vec_input')
            y = Dense(num_feature_vectors * 4, activation='relu', name='feature1')(feature_inputs)
            y = Dense(num_feature_vectors * 8, activation='relu', name='feature2')(y)
            y = Dense(num_feature_vectors * 16, activation='relu', name='feature3')(y)
            z = Concatenate(axis=1)([x, y])
        
        z = Flatten(name='flatten2')(z)
        z = Dense(100, activation='relu', name = 'dense1')(z)
        z = Dropout(drop)(z)
        z = Dense(50, activation='relu', name = 'dense2')(z)
        z = Dropout(drop)(z)      
        z = Dense(25, activation='linear', name = 'dense3')(z)
        z = Dropout(drop)(z)
        

        outputs = Dense(num_outputs, activation='linear', name='output_layer')(z)
            
        if num_feature_vectors > 0:
            model = Model(inputs=[inputs, feature_inputs], outputs=[outputs])
        else:
            model = Model(inputs=[inputs], outputs=[outputs])
        
        return model

class Keras_2D_FULL_HOUSE(Component):
    '''
    Inputs: image, current speed, track localization, track offset
    Outputs: steering, best speed
    '''
    def __init__(self, input_shape, num_outputs, num_feature_vectors = 0):
        pass
    
    @staticmethod
    def get_model(input_shape):
        inputs = Input(shape=input_shape, name='img_input')
        
        drop = 0.1

        x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2),activation='relu', name='conv1')(inputs)
        x = Dropout(drop)(x)
        x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2),activation='relu', name='conv2')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv3')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv4')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv5')(x)
        x = Dropout(drop)(x)
        
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv6')(x)
        x = Dropout(drop)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1),activation='relu', name='conv7')(x)
        x = Dropout(drop)(x)
        x = Flatten(name='flatten1')(x)



        feature_inputs = Input(shape=(1,), name='feature_vec_input')
        y = Embedding(2, 16)(feature_inputs)
        y = Dense(16, activation='relu', name='feature1')(y)
        y = Dense(32, activation='relu', name='feature2')(y)
        y = Dense(64, activation='relu', name='feature3')(y)
        y = tf.squeeze(y, axis=1)

        x = Concatenate(axis=1)([x, y])
      
        z = Flatten(name='flatten2')(x)
        z = Dense(100, activation='relu', name = 'dense1')(z)
        z = Dropout(drop)(z)
        z = Dense(50, activation='relu', name = 'dense2')(z)
        z = Dropout(drop)(z)      
        z = Dense(25, activation='linear', name = 'dense3')(z)
        z = Dropout(drop)(z)
        
        outputs = Dense(2, activation='linear', name='output')(z)
            
        model = Model(inputs=[inputs, feature_inputs], outputs=[outputs,])
        
        return model

class KerasResNetLSTM:
    '''
    Inputs: image, current speed
    Outputs: steering, throttle
    '''
    def __init__(self, input_shape):
        pass

    @staticmethod
    def get_model(input_shape, embedding_size, batch_size):
        img_input = Input(name='img_input', batch_input_shape=(batch_size, *input_shape))
        resize = Resizing(224, 224)
        current_spd_input = Input(shape=(1,), name='current_spd_input')
        encoder = ResNet50(include_top=True, weights='imagenet', classifier_activation='linear')
        fc = Dense(embedding_size, activation='linear', name='encoder_out')
        for layer in encoder.layers: layer.trianable = False
        embed = Embedding(100, int(embedding_size/2))
        encoder.layers[-1].trainable = True
        decoder1 = LSTM(512, stateful=True, name='decoder')
        fc1 = Dense(100, activation='relu', name='dense1')
        fc2 = Dense(50, activation='relu', name='dense2')
        fc3 = Dense(2, activation='linear', name='dense3')

        x = resize(img_input)
        x = encoder(x)
        x = fc(x)
        y = embed(current_spd_input)
        x = tf.expand_dims(x, axis=1)
        x = Concatenate(axis=2)([x, y])
        x= decoder1(x)
        x = fc1(x)
        x = fc2(x)
        outputs = fc3(x)

        model = Model(inputs=[img_input, current_spd_input], outputs=[outputs])
        
        return model

class DonkeyDataLoader(DataLoader):
    def __init__(self, *paths):
        DataLoader.__init__(self, *paths)
    def get_img_name(self, idx):
        return f'{idx}_cam-image_array_.jpg'

    def get_record_name(self, idx):
        return f'record_{idx}.json'

    def get_labels_from_record(self, record={}):
        return np.asarray((record['user/angle'], record['user/throttle']))

    def get_features_from_record(self,record={}):
        '''Any additional features are we looking for?'''
        # return record['gym/speed'], record['gym/cte']
        return None
    
class SpeedFeatureDataLoader(DataLoader):
    def __init__(self, *paths):
        DataLoader.__init__(self, *paths)
    def get_features_from_record(self,record={}):
        '''Any additional features are we looking for?'''
        return np.asarray((record['gym/speed'],))
    
class SpeedCtlDataLoader(DataLoader):
    def __init__(self, mean, offset, *paths):
        DataLoader.__init__(self, *paths)
        self.mean = mean
        self.offset = offset

    def get_labels_from_record(self, record={}):
        return np.asarray((record['mux/steering'], (record['gym/speed'] / self.mean) - self.offset)) # Adjust the input range to be [0, 1]

class SpeedCtlBreakIndicationDataLoader(DataLoader):
    def __init__(self, *paths):
        DataLoader.__init__(self, *paths)

    def get_features_from_record(self,record={}):
        '''Any additional features are we looking for?'''
        return np.asarray((record['loc/break_indicator']))

    def get_labels_from_record(self, record={}):
        return np.asarray((record['mux/steering'], (record['gym/speed'] / 10.0) - 1.0)) # Adjust the input range to be [0, 1]

class LocalizationDemoDataLoader(DataLoader):
    def __init__(self, *paths):
        DataLoader.__init__(self, *paths)
    def get_img_name(self, idx):
        return f'record_{idx}.png'

    def get_record_name(self, idx):
        return f'record_{idx}.json'

    def get_labels_from_record(self, record={}):
        return np.asarray((record['x'] / 20, record['y'] / 20, record['orientation'] / 360), dtype=np.float16) # Adjust the input range to be [0, 1]
    
class FullHouseDataLoader(DataLoader):
    def __init__(self, *paths):
        DataLoader.__init__(self, *paths)
    
    def get_features_from_record(self,record={}):
        '''Any additional features are we looking for?'''
        return np.asarray((record['loc/break_indicator'],))
    
    def get_labels_from_record(self, record={}):
        return np.asarray((record['mux/steering'], (record['gym/speed'] / 10) - 1.0))

    def load(self, train_val_split = 0.8, batch_size = 128):
        print ('Loading records...')
        for data_path in self.paths:
            i = 1
            while True:
                try:
                    # Obtain img as array
                    img_path = path.join(data_path, self.get_img_name(i))
                    img_arr = np.asarray(Image.open(img_path),dtype=np.float32)

                    # Obtain labels and feature vectors as arrays
                    record_path = path.join(data_path, self.get_record_name(i))
                    record={}
                    with open(record_path) as f:
                        record= json.load(f)
                    labels = np.asarray(self.get_labels_from_record(record),dtype=np.float32)
                    feature_vectors = np.asarray(self.get_features_from_record(record), dtype=np.float32)
                    self.dataset.append((img_arr, feature_vectors, labels))
                    # print (labels)
                    i += 1
                except FileNotFoundError:
                    # print (f'Loaded {i-1} records in {data_path}')
                    break

        print (f'Loaded {len(self.dataset)} records.')

    
    def split_train_val(self, split = 0.8, batch_size = 64):
        assert 0 < split <= 1
        from sklearn.model_selection import train_test_split
        train_set, val_set = train_test_split(self.dataset, train_size = split)

        train_examples = []
        train_example_vecs = []
        train_labels = []

        val_examples = []
        val_example_spds = []
        val_example_vecs = []
        val_labels = []
        
        for data in train_set:
            train_examples.append(data[0])
            train_example_vecs.append(np.asarray(data[1][0]))
            train_labels.append(data[2])

        for data in val_set:
            val_examples.append(data[0])
            val_example_vecs.append(np.asarray(data[1][0]))
            val_labels.append(data[2])

        train_examples = np.stack(train_examples, axis=0)
        train_labels = np.stack(train_labels, axis=0)
        train_example_vecs = np.stack(train_example_vecs, axis=0)

        val_examples = np.stack(val_examples, axis=0)
        val_labels = np.stack(val_labels, axis=0)
        val_example_vecs = np.stack(val_example_vecs, axis=0)

        self.train_dataset = tf.data.Dataset.from_tensors(((train_examples, train_example_vecs), train_labels))
        self.val_dataset = tf.data.Dataset.from_tensors(((val_examples,val_example_vecs), val_labels)) 

class LSTMDataLoader(DataLoader):
    def __init__(self, *paths):
        DataLoader.__init__(self, *paths)

    def shuffle_batch(self, batch_size):
        # No actual shuffle. just batching
        # Batch the data into short sequences of 100 images
        self. train_dataset_batch = self.train_dataset.batch(batch_size, drop_remainder=True)
        self.val_dataset_batch = self.val_dataset.batch(batch_size, drop_remainder=True)

        

    def split_train_val(self, split = 0.8, batch_size = 64):
        assert 0 < split <= 1
        train_num = int(len(self.dataset) * split)
        train_set = self.dataset[:train_num]
        val_set = self.dataset[train_num:]

        train_examples = []
        train_example_vecs = []
        train_labels = []

        val_examples = []
        val_example_vecs = []
        val_labels = []
        
        for data in train_set:
            train_examples.append(preprocess_input(data[0]*255))
            train_example_vecs.append(data[1])
            train_labels.append(data[2])

        for data in val_set:
            val_examples.append(preprocess_input(data[0]*255))
            val_example_vecs.append(data[1])
            val_labels.append(data[2])

        train_examples = self.__stack_and_regroup(train_examples, batch_size)
        train_labels = self.__stack_and_regroup(train_labels, batch_size)
        val_examples = self.__stack_and_regroup(val_examples, batch_size)
        val_labels = self.__stack_and_regroup(val_labels, batch_size)

        train_example_vecs = self.__stack_and_regroup(np.expand_dims(train_example_vecs,axis=1), batch_size)
        val_example_vecs = self.__stack_and_regroup(np.expand_dims(val_example_vecs,axis=1), batch_size)
        self.train_dataset = tf.data.Dataset.from_tensors(((train_examples, train_example_vecs), train_labels)).unbatch()
        self.val_dataset = tf.data.Dataset.from_tensors(((val_examples, val_example_vecs), val_labels)).unbatch()   


    def __stack_and_regroup(self, lis, batch_size):
        print("Regrouping...")
        arr = np.stack(lis, axis=0)
        regroup = None
        output = None
        seq_len = int(len(arr) / batch_size)
        for i in range(batch_size):
            seq = np.expand_dims(arr[i*seq_len:i*seq_len+seq_len], axis=1)
            regroup = np.hstack((regroup, seq)) if regroup is not None else seq
        for row in regroup:
            output = np.vstack((output, row)) if output is not None else row
        return output

    def get_img_name(self, idx):
        return f'img_{idx}.jpg'

    def get_record_name(self, idx):
        return f'record_{idx}.json'

    def get_labels_from_record(self, record={}):
        return record['mux/steering'], record['gym/speed'] # Adjust the input range to be [0, 1]

    def get_features_from_record(self,record={}):
        '''Any additional features are we looking for?'''
        return record['gym/speed']
class LSTMCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.get_layer('decoder').reset_states()
        print("Resetting states")
    def on_test_begin(self, logs=None):
        self.model.get_layer('decoder').reset_states()
        print("Resetting states")

def train(cfg, data_paths, model_path, transfer_path=None, shape=None):
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model_cfg = cfg['ai_model']
    loader = None
    model = None

    model_type = ModelType(model_cfg['model_type'])
    if shape is not None:
        input_shape = int(shape[0]), int(shape[1]), 3
    else:
        input_shape = calc_input_shape(cfg)
    batch_size = model_cfg['batch_size']
    print (f"Input shape: {input_shape}")

    if model_type == ModelType.CNN_2D:
        loader = DataLoader(*data_paths)
        model = Keras_2D_CNN.get_model(input_shape=input_shape, num_outputs=2, num_feature_vectors=0)
    elif model_type == ModelType.CNN_2D_SPD_FTR:
        loader = SpeedFeatureDataLoader(*data_paths)
        model = Keras_2D_CNN.get_model(input_shape=input_shape, num_outputs=2, num_feature_vectors=1)
    elif model_type == ModelType.CNN_2D_SPD_CTL:
        loader = SpeedCtlDataLoader(cfg['speed_control']['train_speed_mean'], cfg['speed_control']['train_speed_offset'], *data_paths)
        model = Keras_2D_CNN.get_model(input_shape=input_shape, num_outputs=2, num_feature_vectors=0)
    elif model_type == ModelType.CNN_2D_FULL_HOUSE:
        loader = FullHouseDataLoader(*data_paths)
        model = Keras_2D_FULL_HOUSE.get_model(input_shape=input_shape)
    elif model_type == ModelType.LSTM:
        loader = LSTMDataLoader(*data_paths)
        model = KerasResNetLSTM.get_model(input_shape, model_cfg['embedding_size'], batch_size)
    elif model_type == ModelType.CNN_2D_SPD_CTL_BREAK_INDICATION:
        loader = SpeedCtlBreakIndicationDataLoader(*data_paths)
        model = Keras_2D_CNN.get_model(input_shape=input_shape, num_outputs=2, num_feature_vectors=1)

    if transfer_path is not None:
        model = load_model(transfer_path)
    loader.load(batch_size=batch_size)
    loader.split_train_val(split=0.8, batch_size=batch_size)
    loader.shuffle_batch(batch_size)
    model.compile(optimizer=optimizers.Adam(lr=model_cfg['learning_rate']), loss='mse')
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='auto', verbose=1, save_freq='epoch')
    ]

    if model_cfg['early_stop']:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=model_cfg['early_stop_patience']))
    if model_type == ModelType.LSTM:
        callbacks.append(LSTMCallback())

    model.fit(loader.train_dataset_batch, epochs=model_cfg['max_epoch'], validation_data=loader.val_dataset_batch, callbacks=callbacks)
    print(f'Finished training. Best model saved to {model_path}.')
    # model.save(model_path)

def calc_input_shape(cfg):
    cam_cfg = cfg['cam']
    width = cam_cfg['img_w']
    height = cam_cfg['img_h']
    channel = 1 if cam_cfg['img_format'] == 'GREY' else 3

    if cfg['img_preprocessing']['enabled']:
        t, b, l, r = cfg['img_preprocessing']['crop']
        width -= l + r
        height -= t + b
    return height, width, channel




