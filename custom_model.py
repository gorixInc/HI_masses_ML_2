# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:31:03 2021

@author: Gordei
"""
import random

import tensorflow as tf
import numpy as np
from random import sample, shuffle
from sklearn.preprocessing import scale, StandardScaler
from math import floor


class Dense(tf.Module):  # Denesly connected leyer
    # Note: No need for `in_features`
    def _linear_act(x):
        return x

    def __init__(self, out_features, act_func=_linear_act, name='Dense'):
        super().__init__(name=name)
        self.desc = name
        self.is_built = False
        self.out_features = out_features  # Number of outputs
        self.act_func = act_func
        self.r_seed = random.randint(1, 99999)

    def __call__(self, x):
        # Create variables on first call.
        if not self.is_built:
            #self.w = tf.Variable(tf.random.normal(
            #    [x.shape[-1], self.out_features], seed=self.r_seed),
            #    name='w')
            self.w = tf.Variable(tf.zeros([x.shape[-1],self.out_features]),
                name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
            self.is_built = True

        y = tf.matmul(x, self.w) + self.b
        return self.act_func(y)

    
class Dropout(tf.Module):
    def __init__(self, rate, name='Dropout'):
        super().__init__(name=name)
        self.desc = 'Dropout {}'.format(rate)
        self.rate = rate
        self.r_seed = random.randint(1, 9999999)
     
    def __call__(self, x):
        return tf.nn.dropout(x, self.rate, seed=self.r_seed)
    

#class Batch_norm(tf.Module):
#    def __init__

    
class Model(tf.Module):
    def reset(self):
        for layer in self.layers:
            if self.r_seed == -1:
                layer.r_seed = random.randbytes(4)
            if isinstance(layer, Dense):
                layer.is_built = False

    def __init__(self, r_seed=-1, layers=[], name=None):
        super().__init__(name=name)
        self.layers = layers
        self.r_seed = r_seed
        self.layer_dict = {}

        for layer in self.layers:
            layer.r_seed = self.r_seed
            if self.r_seed == -1:
                layer.r_seed = random.randbytes(4)
            desc = layer.desc
            if desc not in self.layer_dict:
                self.layer_dict[desc] = 0
            self.layer_dict[desc] += 1

    # Feed forward on call
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_description(self):
        desc = 'Model with '
        for name in self.layer_dict:
            desc += ('{}: {}; '.format(name, self.layer_dict[name]))
        return desc

    def predict(self, x):
        rate = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if layer.name == 'Dropout':
                rate = layer.rate
            else:
                x = layer.act_func(tf.matmul(x, layer.w*(1-rate)) + layer.b)
                rate = 0

        return x


def _loss(model, inputs, target_y):
    pred_y = model(inputs)
    return tf.reduce_mean(tf.square(target_y - pred_y))


def _grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = _loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def _get_batches(X, Y, batch_size, scale_batch=False):
    n_data = len(X)
    # Shuffle data
    c = list(zip(X, Y))
    shuffle(c)
    X, Y = zip(*c)
    # Splitting into batches
    X, Y = np.array(X).astype('float32'), np.array(Y).astype('float32')
    x, y = [], []
    for i in range(0, n_data - batch_size, batch_size):
        batch_x = X[i:i + batch_size]
        batch_y = Y[i:i + batch_size]
        if scale_batch:
            scaler = StandardScaler()
            batch_x = scaler.fit_transform(batch_x)
            #batch_y = np.reshape(batch_y, (1, -1))
            #scaler.fit(batch_y)
            #batch_y = scaler.transform(batch_y)
            #print(batch_y)
        x.append(batch_x)
        y.append(batch_y)
    return x, y


def train(model,train_set,valid_set,test_set,
          epochs=10,
          batch_size=1,
          optimiser=tf.optimizers.Adam(0.01),
          print_every=100):
    
    X, Y = train_set[0], train_set[1]
    x_valid, y_valid = valid_set[0], valid_set[1]
    x_test, y_test = test_set[0], test_set[1]
    train_loss_results = []
    valid_loss_results = []
    test_loss_results = []

    for epoch in range(epochs):
        ##STATS##
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_loss_valid = tf.keras.metrics.Mean()
        epoch_loss_test = tf.keras.metrics.Mean()
        ##STATS##
        # Creating new batches for every epoch
        x, y = _get_batches(X, Y, batch_size)
        for i in range(0, len(x)):  # For all each batch
            loss_value, grads = _grad(model, x[i], y[i])
            ##STATS##
            epoch_loss_avg.update_state(loss_value)
            epoch_loss_valid.update_state(_loss(model, x_valid, y_valid))
            epoch_loss_test.update_state(_loss(model, x_test, y_test))
            ##STATS##
            optimiser.apply_gradients(zip(grads, model.trainable_variables))

        train_loss_results.append(epoch_loss_avg.result().numpy())
        valid_loss_results.append(epoch_loss_valid.result().numpy())
        test_loss_results.append(epoch_loss_test.result().numpy())
        if ((epoch + 1) % print_every == 0):
            print("Epoch: {}; Loss: {}".format(
                epoch + 1,
                epoch_loss_avg.result().numpy()))
    return (train_loss_results, valid_loss_results, test_loss_results)