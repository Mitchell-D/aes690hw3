from pathlib import Path
from random import random
import pickle as pkl

import numpy as np
import os
import sys
import h5py
import json
import random as rand
from list_feats import dynamic_coeffs,static_coeffs

#import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking,Reshape,ReLU,Conv1D,Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

def_dense_kwargs = {
        "activation":"sigmoid",
        "use_bias":True,
        "bias_initializer":"zeros",
        "kernel_initializer":"glorot_uniform",
        "kernel_regularizer":None,
        "bias_regularizer":None,
        "activity_regularizer":None,
        "kernel_constraint":None,
        "bias_constraint":None,
        }

def load_config(model_dir):
    """
    Load the configuration JSON associated contained in a model directory
    """
    model_name = model_dir.name
    return json.load(model_dir.joinpath(f"{model_name}_config.json").open("r"))

def load_csv_prog(model_dir):
    """
    Load the per-epoch metrics from a tensorflow CSVLogger file as a dict.
    """
    cfg = load_config(model_dir)
    csv_path = model_dir.joinpath(f"{cfg['model_name']}_prog.csv")
    csv_lines = csv_path.open("r").readlines()
    csv_lines = list(map(lambda l:l.strip().split(","), csv_lines))
    csv_labels = csv_lines.pop(0)
    csv_cols = list(map(
        lambda l:np.asarray([float(v) for v in l]),
        zip(*csv_lines)))
    return dict(zip(csv_labels, csv_cols))

def get_dense_stack(name:str, layer_input:Layer, node_list:list,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """
    Simple stack of dense layers with optional dropout and batchnorm
    """
    dense_kwargs = {**def_dense_kwargs.copy(), **dense_kwargs}
    l_prev = layer_input
    for i in range(len(node_list)):
        l_new = Dense(
                units=node_list[i],
                **dense_kwargs,
                name=f"{name}_dense_{i}"
                )(l_prev)
        if batchnorm:
            l_new = BatchNormalization(name=f"{name}_bnorm_{i}")(l_new)
        if dropout_rate>0.0:
            l_new = Dropout(dropout_rate)(l_new)
        l_prev = l_new
    return l_prev

def basic_dense(name:str, node_list:list, num_window_feats:int,
        num_horizon_feats:int, num_static_feats:int, num_pred_feats:int,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """
    Dense layer next-step predictor model that simply appends window and
    horizon feats as the input, and outputs the prediction features for the
    next time step. The only reason there's a distinction between window and
    horizon features is to conform to the input tensor style used by others.
    """
    w_in = Input(shape=(1,num_window_feats,), name="in_window")
    h_in = Input(shape=(1,num_horizon_feats,), name="in_horizon")
    s_in = Input(shape=(num_static_feats,), name="in_static")
    mod_in = Reshape(target_shape=(1,num_static_feats))(s_in)
    all_in = Concatenate(axis=-1)([w_in,h_in,mod_in])
    dense = get_dense_stack(
            name=name,
            node_list=node_list,
            layer_input=all_in,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=dense_kwargs,
            )
    output = Dense(units=num_pred_feats,
            activation="linear",name="output")(dense)
    inputs = {"window":w_in,"horizon":h_in,"static":s_in}
    model = Model(inputs=inputs, outputs=[output])
    return model

def get_sample_generator(train_h5s,val_h5s,window_size,horizon_size,
        window_feats,horizon_feats,pred_feats,static_feats):
    """
    Returns generators which provide window, horizon, and static data
    as features, and prediction data as labels by subsetting a larger
    sequence per-sample.
    """
    ## Nested output signature for gen_hdf5_sample
    out_sig = ({
        "window":tf.TensorSpec(
            shape=(window_size,len(window_feats)), dtype=tf.float64),
        "horizon":tf.TensorSpec(
            shape=(horizon_size,len(horizon_feats)), dtype=tf.float64),
        "static":tf.TensorSpec(
            shape=(len(static_feats),), dtype=tf.float64)
        },
        tf.TensorSpec(shape=(horizon_size,len(pred_feats)), dtype=tf.float64))

    pos_args = (
            window_size,horizon_size,
            window_feats,horizon_feats,
            pred_feats,static_feats
            )
    gen_train = tf.data.Dataset.from_generator(
            gen_sample,
            args=(train_h5s, *pos_args),
            output_signature=out_sig,
            )
    gen_val = tf.data.Dataset.from_generator(
            gen_sample,
            args=(val_h5s, *pos_args),
            output_signature=out_sig,
            )
    return gen_train,gen_val

if __name__=="__main__":
    sample_dir = Path("/rstor/mdodson/thesis")
    h5s_val = [sample_dir.joinpath("shuffle_2018.h5").as_posix()]
    h5s_train = [sample_dir.joinpath(f"shuffle_{y}.h5").as_posix()
        for y in [2015,2019,2021]]
    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf"]
    pred_feats = ['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200']
    static_feats = ["pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"]

    g = gen_sample(
            h5_paths=h5s_train,
            window_size=24,
            horizon_size=24,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            as_tensor=False,
            )

    #for i in range(10000):
    #    x,y = next(g)
    #    print([x[k].shape for k in x.keys()], y.shape)
    gT,gV = get_sample_generator(
            train_h5s=h5s_train,
            val_h5s=h5s_val,
            window_size=24,
            horizon_size=24,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            )
    batches = [next(g) for i in range(2048)]
    pkl.dump(batches, Path("data/sample/batch_samples.pkl").open("wb"))
