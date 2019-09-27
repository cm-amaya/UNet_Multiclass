import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import argparse

import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
import segmentation_models as sm
from utils import visualize, freeze_session
from data_loader import Dataset, Dataloader
from keras.backend.tensorflow_backend import set_session
from augmentation import *

# Tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#ArgParse
parser = argparse.ArgumentParser(description='UNET for Multiclass Semantic Segmentation')
parser.add_argument('--train_dir', type=str, required = True, help='Train dir - with subdirs images ans masks')
parser.add_argument('--val_dir', type=str, required = True, help='Val dir - with subdirs images ans masks')
parser.add_argument('--result_dir', type=str,default="results", help='Result dir - where the model will be saved')
parser.add_argument('--image_size', type=int,default=320, help='Image size - for cropping the images to nxn images')
parser.add_argument('--image_channels', type=int,default=3, help='Image channels - number of channels of the input image')
parser.add_argument('--padding_size', type=int,default=800, help='Padding size for Val images - must be a multiple of image size')
parser.add_argument('--n_classes', type=int,default=2, help='# of classes - number of classes')
parser.add_argument('--batch_size', type=int,default=2, help='Batch size')
parser.add_argument('--epochs', type=int,default=100, help='# of Epochs')
args = parser.parse_args()

# Directorios
train_dir = args['train_dir']
test_dir = args['val_dir']
result_dir = args['result_dir']

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    
x_train_dir = os.path.join(train_dir, 'images')
y_train_dir = os.path.join(train_dir, 'masks')
x_valid_dir = os.path.join(test_dir, 'images')
y_valid_dir = os.path.join(test_dir, 'masks')

#Model parameters
BACKBONE = 'efficientnetb3'
BATCH_SIZE = args['batch_size']
LR = 0.0001
EPOCHS = args['epochs']

# define network parameters 
activation ='softmax'
n_classes = args['n_classes']
image_size = args['image_size']
image_channels = args['image_channels']
padding_size = args['padding_size']
preprocess_input = sm.get_preprocessing(BACKBONE)

# Dataset for train images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=n_classes, 
    augmentation=get_training_augmentation(imgsize = image_size),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=n_classes, 
    augmentation=get_validation_augmentation(imgsize = padding_size),
    preprocessing=get_preprocessing(preprocess_input),
)

#create model
model = sm.Unet(BACKBONE,classes=n_classes, activation=activation)

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss(class_weights=np.ones(n_classes)) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)


#Dataloaderss
train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, image_size, image_size, image_channels)
assert train_dataloader[0][1].shape == (BATCH_SIZE, image_size, image_size, n_classes)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(result_dir,'best_model.h5'), save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(result_dir,'UNET-Training.png'))
plt.show()

#Save model as Tensorflow frozen graph
frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, result_dir , "best_model.pb", as_text=False)