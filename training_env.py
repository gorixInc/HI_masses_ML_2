import os

from custom_model import Model, Dense, Dropout, train
import pandas as pd
from random import Random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def calc_variation_measure(predictions):
    stds = []
    predictions = np.array(predictions)
    for i in range(len(predictions[0])):
        stds.append(np.std(predictions[:, i]))
    return np.mean(np.array(stds)**2)


class TrainingEnv:
    def __init__(self, models, epoch_number_arr, opt_arr, lr_arr, batch_size_arr, n_runs, data_dir, write_dir, rseed):
        self.random = Random(rseed)
        self.models = models
        self.data_dir = data_dir

        self.main_df = pd.DataFrame()
        self.pred_df = pd.DataFrame()
        self.epoch_number_arr = epoch_number_arr
        self.lr_arr = lr_arr
        self.opt_arr = opt_arr
        self.batch_size_arr = batch_size_arr
        self.n_runs = n_runs

        now = datetime.now()
        self.write_dir = '{}/{}'.format(write_dir, now.strftime('%d-%m-%Y--%H%M%S'))
        os.makedirs(self.write_dir)

    def run(self):
        data_arrs, s_features, logits = self.pre_process_data(self.data_dir)
        for model in self.models:   # For each model
            for i in range(len(self.lr_arr)):  # For each set of parameters
                n_epochs = self.epoch_number_arr[i]
                optimizer = self.opt_arr[i]
                lr = self.lr_arr[i]
                batch_size = self.batch_size_arr[i]
                predictions = []
                losses = []
                for j in range(self.n_runs):  # Each run with current parameters
                    model.reset()
                    train(model, data_arrs[0], data_arrs[1], data_arrs[2], epochs=n_epochs,
                          batch_size=batch_size, optimiser=optimizer(lr), print_every=500)
                    pred_logits = model.predict(s_features)
                    predictions.append(pred_logits)
                    loss_arr = []
                    for data_arr in data_arrs:
                        pred = model.predict(data_arr[0])
                        loss_arr.append(mean_squared_error(pred, data_arr[1]))
                    losses.append(loss_arr)
                var_measure = calc_variation_measure(predictions)
                avg_losses = np.mean(losses, axis=0)
                std_losses = np.std(losses, axis=0)
                metad_dict = {'desc': model.get_description(), 'n_epochs': n_epochs, 'optimiser': optimizer.__name__,
                              'lr': lr, 'batch_size': batch_size}
                row = {'var_measure': var_measure, 'train_loss_avg': avg_losses[0],
                       'train_loss_std': std_losses[0], 'valid_loss_avg': avg_losses[1], 'valid_loss_std': avg_losses[1],
                       'test_loss_avg': avg_losses[2], 'test_loss_std': avg_losses[2]}
                self.main_df = self.main_df.append(metad_dict | row, ignore_index=True)

                predictions = np.array(predictions)[:, :, 0]
                self.pred_df = pd.DataFrame(predictions, columns=['gal_{}'.format(i) for i in range(len(predictions[0]))])
                self.pred_df.insert(0, 'desc', [metad_dict['desc'] for _ in range(len(predictions))])
                self.pred_df.to_csv('{}/preds-{}-{}-{}-{}.csv'.format(self.write_dir, n_epochs, optimizer.__name__, lr,
                                                                      batch_size))
        self.main_df.to_csv('{}/results.csv'.format(self.write_dir))

    def pre_process_data(self, arecibo_path):
        arecibo_df = pd.read_csv(arecibo_path, index_col=[0])
        scaler = StandardScaler()
        imputer = SimpleImputer()

        features = arecibo_df.drop(['Name', 'MHI'], axis=1).values
        logits = arecibo_df['MHI'].values

        features = imputer.fit_transform(features).astype('float32')
        logits = logits.reshape((-1, 1))
        logits = imputer.fit_transform(logits).astype('float32')

        c = list(zip(features, logits))
        self.random.shuffle(c)
        f_shuff, l_shuff = zip(*c)

        n = len(features)
        n_valid = 20
        n_test = 10

        ind_train = n - n_test - n_valid
        ind_valid = n - n_test

        train_arr = [f_shuff[0:ind_train], l_shuff[0:ind_train]]
        valid_arr = [f_shuff[ind_train:ind_valid], l_shuff[ind_train:ind_valid]]
        test_arr = [f_shuff[ind_valid:], l_shuff[ind_valid:]]

        data_arrs = [train_arr, valid_arr, test_arr]

        for i in range(3):
            data_arrs[i][0] = scaler.fit_transform(data_arrs[i][0]).astype('float32')

        s_features = scaler.fit_transform(features).astype('float32')
        return data_arrs, s_features, logits
