# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:15:50 2019

@author: user
"""
import os
import cv2
import numpy as np 
import keras

class Dataset:
    def __init__(self,images_dir,masks_dir,classes=2,augmentation=None,preprocessing=None):
        images = []
        img_list = os.listdir(images_dir)
        #Check images
        for file in img_list:
             try:
                 image = cv2.imread(os.path.join(images_dir,file))
                 mask = cv2.imread(os.path.join(masks_dir,file))
                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                 images.append(file)
             except:
                  print("Cannot load: {}".format(file))
                  continue
        
        self.ids = images
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.classes = classes
        
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in range(self.classes)]
        mask = np.stack(masks, axis=-1).astype('float')
    
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class TestDataset:
    def __init__(self,images_dir,classes=None,augmentation=None,preprocessing=None):
        images = []
        img_list = os.listdir(images_dir)
        for file in img_list:
             try:
                 image = cv2.imread(os.path.join(images_dir,file))
                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                 images.append(file)
             except:
                  print("An exception occurred: {}".format(file))
                  continue
        
        self.ids = images
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.classes = classes
        
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image,self.images_fps[i]
        
    def __len__(self):
        return len(self.ids)
    
class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)