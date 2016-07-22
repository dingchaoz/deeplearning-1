
# Author: Raj Agrawal 

# Builds a 3D spatio-temporal convolutional neural network to detect texting and 
# driving from a video stream 

# Quick Architectural Overview:
# - 3 convolutional layers (ReLu, Dropout, MaxPooling), 2 dense layers
# - Binary Hinge-Loss 
# - Nesterov Momentum update w/ learning rate decaying linearly w/ num. epochs 
# - Early Stopping 

# References: See paper. Special thanks to Daniel Nouri for his tutorial at 
# http://danielnouri.org/notes/category/machine-learning/ (Some code snippets  
# taken directly from this tutorial) 

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

# TODO: Fix up/verify arch. and sizes 
def build_cnn():
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

# Place to modify how to tune momentum and learning rate 
class AdjustVariable(object):
    """
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

# Place to add more variety - specify some proportion in each cat. to alter
# Could preprocess here as well? 
class FlipBatchIterator(BatchIterator):
    """
    """

    # Flip images 
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    # Change intensities values 
    # TODO: Put 3 different types of things based on number 1, 2, 3 (that will
    # be randomly drawn), translate, etc. see famous Krivensky paper  

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1] #FIX: Ours is different matrix struct. 

        # if yb is not None:
        #     # Horizontal flip of all x coordinates:
        #     yb[indices, ::2] = yb[indices, ::2] * -1

        #     # Swap places, e.g. left_eye_center_x -> right_eye_center_x
        #     for a, b in self.flip_indices:
        #         yb[indices, a], yb[indices, b] = (
        #             yb[indices, b], yb[indices, a])

        return Xb, yb

# This gets called whenever finish epoch. If I want to save weights every x epochs
# this is where to do it - maybe set number of max_epochs high since then will just
# early stop 
class EarlyStopping(object):
    """
    """
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        
        #Save weights to s3 every 20 epochs 

        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

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

#Fix but make sure to multiply by different p's b/c dropout 
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3

if __name__ == '__main__':

    # Load data (did not standardize b/c images in 0-256)
    X = np.load('./data/train/images_by_time_mat') 
    Y = np.load('./data/train/labels')

    # Fit model 
    network.fit(X, Y)

    # Save Model 
    with open('network.pickle', 'wb') as f:
        pickle.dump(net3, f, -1)
        pickle.dump(network, './models/3dcnn')
