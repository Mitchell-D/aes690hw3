from pathlib import Path

from random import random
import pickle as pkl
import json
import numpy as np
import os
import sys

#import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#'''
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus):
    tf.config.experimental.set_memory_growth(gpus[0], True)
#'''

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

def feedforward(name:str, node_list:list, num_inputs:int, num_outputs:int,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """
    Dense layer next-step predictor model that simply appends window and
    horizon feats as the input, and outputs the prediction features for the
    next time step. The only reason there's a distinction between window and
    horizon features is to conform to the input tensor style used by others.
    """
    ff_in = Input(shape=(num_inputs,), name="input")
    dense = get_dense_stack(
            name=name,
            node_list=node_list,
            layer_input=ff_in,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=dense_kwargs,
            )
    output = Dense(units=num_outputs, activation="linear",name="output")(dense)
    model = Model(inputs=ff_in, outputs=output)
    return model

def gen_noisy(X, Y, noise_pct=0, noise_stdev=0, mask_val=9999.,
              feat_probs:np.array=None, shuffle=True, rand_seed=None):
    """
    Generates (X, Y, sample_weight) triplets for training a model with maskable
    feature values. The percentage of masked values determines the weight.

    The feature dimension is always assumed to be the last one. For example...

    (S,F) is a dataset with F features measured in S samples
    (S,L,F) is a dataset with S sequences of L members, each having F features.
    (S,M,N,F) is a 2d (M,N) dataset with F features and S sample instances.

    :@param X: (S,F) array of S samples of F features.
    :@param Y: (S,P) array of S samples of P predicted values.
    :@param noise_pct: mean percentage of feature values to mask.
    :@param noise_stdev: standard deviation of feature values to mask.
    :@param mask_val: Value to substitute for the mask.
    :@param feat_probs: Array with the same size as number of features, which
        provides the relative probabilities of each feature being selected to
        be masked. Uniform by default.
    :@param shuffle: if True, randomly shuffles samples along the first axis.
    """
    num_feats = X.shape[-1]
    num_samples = X.shape[0]
    ## Make sure the feature probability array is formatted correctly
    if feat_probs is None:
        feat_probs = np.full(shape=(num_feats,), fill_value=1.)
    else:
        assert np.array(feat_probs).squeeze().shape == (num_feats,)
        assert np.all(feat_probs >= 0.)
    ## Shuffle along the sample axis, if requested.
    if shuffle:
        rand_idxs = np.arange(num_samples)
        np.random.seed(rand_seed)
        np.random.shuffle(rand_idxs)
        X = X[rand_idxs]
        Y = Y[rand_idxs]
    ## Preserve ratios and convert to probabilities summing to 1
    feat_probs = feat_probs / np.sum(feat_probs)
    ## Pick a number of features to mask according to a distribution of
    ## percentages saturating at 0 and 1, parameterized by the user.
    noise_dist = np.random.normal(noise_pct,noise_stdev,size=num_samples)
    mask_count = np.rint(np.clip(noise_dist,0,1)*num_feats).astype(int)
    feat_idxs = np.arange(num_feats).astype(int)
    ##(!!!) The feature dimension is always assumed to be the final one (!!!)##
    for i in range(num_samples):
        ## Choose indeces that will be masked in each sample
        mask_idxs = np.random.choice(
                feat_idxs,
                size=mask_count[i],
                replace=False,
                #p=feat_probs,
                )
        X[i,...,mask_idxs] = mask_val
    weights = 1-mask_count/num_feats
    for i in range(num_samples):
        yield (X[i], Y[i], weights[i])

def get_generators(X, Y, tv_ratio=.8, noise_pct=0, noise_stdev=0,
                   mask_val=9999., feat_probs:np.array=None, shuffle=True,
                   rand_seed=None, dtype=tf.float64):
    """
    Get training and validation dataset generators for 3-tuples like (x,y,w)
    for input x, true output y, and sample weight w. Optionally use a random
    masking strategy to corrupt a subset of the features, adjusting the sample
    weight proportional to the percentage of values that were masked.

    :@param X: Inputs as an array with shape like (S, ... ,F)
        for S samples and F input features. May be 2+ dimensional.
    :@param Y: Truth outputs as an array with shape like (S, ... ,P)
        for S samples and P predicted features. May be 2+ dimensional.
    :@param tv_ratio: Ratio of training samples to total (sans validation)
    :@param noise_pct: mean percentage of feature values to mask.
    :@param noise_stdev: standard deviation of feature values to mask.
    :@param mask_val: Value to substitute for the mask.
    :@param feat_probs: Array with the same size as number of features, which
        provides the relative probabilities of each feature being selected to
        be masked. Uniform by default.
    :@param shuffle: if True, randomly shuffles samples along the first axis.
    """
    num_samples = X.shape[0]
    num_feats = X.shape[-1]
    assert Y.shape[0] == num_samples
    if shuffle:
        rand_idxs = np.arange(num_samples)
        np.random.seed(rand_seed)
        np.random.shuffle(rand_idxs)
        X = X[rand_idxs]
        Y = Y[rand_idxs]
        split_idx = np.array([int(tv_ratio*num_samples)])
    Tx,Vx = np.split(X, split_idx)
    Ty,Vy = np.split(Y, split_idx)
    out_sig = tf.TensorSpec(shape=Y.shape, dtype=dtype)
    #'''
    out_sig = (
            tf.TensorSpec(shape=Tx.shape[1:],dtype=dtype),
            tf.TensorSpec(shape=Ty.shape[1:], dtype=dtype),
            tf.TensorSpec(shape=tuple(), dtype=dtype),
            )
    #'''
    if feat_probs is None:
        feat_probs = np.full(shape=(num_feats,), fill_value=1.)
    gen_train = tf.data.Dataset.from_generator(
            gen_noisy,
            args=(Tx,Ty,noise_pct,noise_stdev,mask_val, feat_probs, shuffle),
            output_signature=out_sig,
            )
    gen_val = tf.data.Dataset.from_generator(
            gen_noisy,
            args=(Vx,Vy,noise_pct,noise_stdev,mask_val, feat_probs, shuffle),
            output_signature=out_sig,
            )
    return gen_train,gen_val

if __name__=="__main__":
    """ Directory with sub-directories for each model. """
    data_dir = Path("data")
    model_parent_dir = data_dir.joinpath("models")
    asos_al_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
    asos_ut_path = data_dir.joinpath("UT_ASOS_Mar_2023.csv")

    config = {
            "model_name":"ffm-4",
            "input_feats":["tmpc","dwpc","relh","sknt",
                "mslp","p01m","gust","feel"],
            "output_feats":["romps_LCL_m","lcl_estimate"],
            "source_csv":asos_al_path.as_posix(),
            "batch_size":32,
            "batch_buffer":4,
            "dropout_rate":0.2,
            "batchnorm":True,
            "dense_kwargs":{"activation":"sigmoid"},
            "node_list":[256,256,128,128,64,64,64,32,32,32,16,16],
            "loss":"mse",
            "metrics":["mse", "mae"],
            "max_epochs":128, ## maximum number of epochs to train
            "train_steps_per_epoch":16, ## number of batches per epoch
            "val_steps_per_epoch":3, ## number of batches per validation
            "val_frequency":1, ## epochs between validation
            "learning_rate":1e-5,
            "early_stop_metric":"val_mse", ## metric evaluated for stagnation
            "early_stop_patience":30, ## number of epochs before stopping
            "mask_pct":.35,
            "mask_pct_stdev":.6,
            "mask_val":9999.,
            "train_val_ratio":.65,
            "mask_feat_probs":None,
            "notes":"Huge model;strong masking;sigmoid activation;dropout;slower learning rate",
            }

    from preprocess import preprocess
    data_dict = preprocess(
            asos_csv_path=asos_al_path,
            input_feats=config["input_feats"],
            output_feats=config["output_feats"],
            normalize=True,
            )

    """ Initialize the model according to the configuration """
    model = feedforward(
            name=config["model_name"], ## feedforward
            node_list=config["node_list"],
            num_inputs=len(config["input_feats"]),
            num_outputs=len(config["output_feats"]),
            batchnorm=config["batchnorm"],
            dropout_rate=config["dropout_rate"],
            dense_kwargs=config["dense_kwargs"],
            )

    """ Create and add some information to the dedicated model directory """
    model_dir = model_parent_dir.joinpath(config["model_name"])
    assert not model_dir.exists()
    model_dir.mkdir()
    model_json_path = model_dir.joinpath(f"{config['model_name']}_config.json")
    model_json_path.open("w").write(json.dumps(config,indent=4))

    ## Write a model summary to stdout and to a file
    model.summary()
    summary_path = model_dir.joinpath(config["model_name"]+"_summary.txt")
    with summary_path.open("w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    """ Compile and fit the model """
    ## Compile the model
    model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config["learning_rate"]),
            loss=config["loss"],
            metrics=config["metrics"],
            )
    ## Define callbacks for model progress tracking
    gT,gV = get_generators(
            X=data_dict["X"],
            Y=data_dict["Y"],
            tv_ratio=config["train_val_ratio"],
            noise_pct=config["mask_pct"],
            noise_stdev=config["mask_pct_stdev"],
            mask_val=config["mask_val"],
            feat_probs=config["mask_feat_probs"],
            shuffle=True,
            dtype=tf.float64
            )
    callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=config["early_stop_metric"],
                patience=config["early_stop_patience"]
                ),
            tf.keras.callbacks.ModelCheckpoint(
                monitor="val_loss",
                save_best_only=True,
                filepath=model_dir.joinpath(
                    config['model_name']+"_{epoch}_{val_mae:.02f}.hdf5")
                ),
            tf.keras.callbacks.CSVLogger(
                model_dir.joinpath(f"{config['model_name']}_prog.csv"),
                )
            ]
    ## Train the model on the generated tensors
    hist = model.fit(
            gT.batch(config["batch_size"]).prefetch(config["batch_buffer"]),
            epochs=config["max_epochs"],
            ## Number of batches to draw per epoch. Use full dataset by default
            #steps_per_epoch=config["train_steps_per_epoch"],
            validation_data=gV.batch(config["batch_size"]
                                     ).prefetch(config["batch_buffer"]),
            ## batches of validation data to draw per epoch
            #validation_steps=config["val_steps_per_epoch"],
            ## Number of epochs to wait between validation runs.
            validation_freq=config["val_frequency"],
            callbacks=callbacks,
            verbose=2,
            )
    model.save(model_dir.joinpath(config["model_name"]+".keras"))
