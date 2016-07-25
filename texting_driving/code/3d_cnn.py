# Author: Raj Agrawal 

# Builds a 3D spatio-temporal convolutional neural network to detect texting and 
# driving from a video stream 

# Quick Architectural Overview:
# - 3 convolutional layers (ReLu, Dropout, MaxPooling), 2 dense layers
# - Binary Hinge-Loss 
# - Nesterov Momentum update w/ learning rate decaying linearly w/ num. epochs 
# - Early Stopping 

# References: See paper. Special thanks to Daniel Nouri for his tutorial at 
# http://danielnouri.org/notes/category/machine-learning/ 

# TODO: check this and prev code for say 20/5 = 4.0 convert to int 

from __future__ import division 

import lasagne
import theano
import numpy as np
import skimage.transform
from skimage import color
import cPickle as pickle

from lasagne.layers.shape import PadLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.objectives import binary_hinge_loss
from lasagne.updates import nesterov_momentum
from lasagne import layers

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

from random_image_generator import * 

# TODO: Fix up/verify arch. and sizes 
def build_cnn(): #Change take as input image size 
    """
    Builds a 3D spatio-temporal CNN 
    Returns
    -------
    dict
        A dictionary containing the network layers, where the output layer is at key 'output'
    """
    net = {}
    net['input'] = InputLayer((None, 1, 10, 108, 192))

    # ----------- 1st Conv layer group ---------------
    net['conv1a'] = Conv3DDNNLayer(net['input'], 32, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False)
    net['pool1']  = MaxPool3DDNNLayer(net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))
    net['dropout1'] = DropoutLayer(net['pool1'], p=.1)

    # ------------- 2nd Conv layer group --------------
    net['conv2a'] = Conv3DDNNLayer(net['dropout1'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool2']  = MaxPool3DDNNLayer(net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))
    net['dropout2'] = DropoutLayer(net['pool2'], p=.3)

    # ----------------- 3rd Conv layer group --------------
    net['conv3a'] = Conv3DDNNLayer(net['dropout2'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool3']  = MaxPool3DDNNLayer(net['conv3a'],pool_size=(2,2,2),stride=(2,2,2))
    net['dropout3'] = DropoutLayer(net['pool3'], p=.5)

    # ----------------- Dense Layers -----------------
    net['fc4']  = DenseLayer(net['dropout3'], num_units=256,nonlinearity=lasagne.nonlinearities.rectify)
    net['dropout4'] = DropoutLayer(net['fc4'], p=.5)
    net['fc5']  = DenseLayer(net['dropout4'], num_units=256,nonlinearity=lasagne.nonlinearities.rectify)

    # ----------------- Output Layer -----------------
    net['output']  = DenseLayer(net['fc5'], num_units=256, nonlinearity=None)

    return net
 
class AdjustVariable(object):
    """
    Class controlling how to tune the momentum and learning rate
    """
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
    """
    Note: Did not alter intensity values b/c already did that (1.5x increase of
          raw data size by artifically adding modified intensity values)
    """ 
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Distort half of the images in this batch at random:
        bs = Xb.shape[0]
        num_changes = int(bs * .75)
        indices = np.random.choice(bs, num_changes, replace=False)
        distorts_per_cat = int(len(indices) / 4)
        flip_indcs = indices[0:distorts_per_cat]
        flip_indcs2 = indices[distorts_per_cat:(2*distorts_per_cat)]
        flip_indcs3 = indices[(2*distorts_per_cat):(3*distorts_per_cat)]
        rotate_indcs = indices[(3*distorts_per_cat):(4*distorts_per_cat)]
        Xb[flip_indcs] = Xb[flip_indcs, :, ::-1, :] #Verify good flip 
        Xb[flip_indcs2] = Xb[flip_indcs2, :, :, ::-1] 
        Xb[flip_indcs3] = Xb[flip_indcs3, :, ::-1, ::-1]
        for i in rotate_indcs:
            Xb[i, :, :, :] = random_image_generator(Xb[i, :, :, :])
        return Xb, yb

# maybe set number of max_epochs high since then will just
class EarlyStopping(object):
    """
    """
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.num_epochs = 0 

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        self.num_epochs += 1 

        # Save weights every 20 epochs to server (transport to s3 eventually)
        if self.num_epochs % 20 == 0:
            weights = nn.get_all_params_values()
            weight_path = '../data/train/weights/cnn' + self.num_epochs
            with open(weight_path, 'wb') as f:
                pickle.dump(weights, f, -1)

        # Update pointer if there are better weights 
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        
        # Seems like we might be starting to overfit 
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

# Fix but make sure to multiply by different p's b/c dropout 
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3
 
# Build CNN
layers = build_cnn()

network = NeuralNet(
    layers=layers['output'],
    max_epochs=5000,
    
    update=nesterov_momentum,
    objective_loss_function=binary_hinge_loss,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    
    regression=False,
    batch_iterator_train=FlipBatchIterator(batch_size=128, shuffle=False) #Data already shuffled 
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200)
        ],

    verbose=1
)

if __name__ == '__main__':

    # Load data (did not standardize b/c images in 0-256)
    X = np.load('./data/train/images_by_time_mat.npy') 
    Y = np.load('./data/train/labels.npy')

    # Fit model 
    network.fit(X, Y)

    # Save Model 
    with open('../model/network.pickle', 'wb') as f:
        pickle.dump(network, f, -1)
