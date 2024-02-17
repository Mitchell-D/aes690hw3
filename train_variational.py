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

import model_methods as mm

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

def train_variational(X, Y, model_config, model_dir):
    """
    Train a variational model according to a config dictionary with the keys:

    "model_name": unique
    "input_feats": subset of ASOS features to use as network inputs
    "output_feats": subset of ASOS features to predict with the network
    "latent_size": Number of posterior-approximating gaussian distriubtions
    "source_csv": string path to a ASOS monthly csv file
    "batch_size": int minibatch size (samples per weight update)
    "batch_buffer": int num of batches to preload in memory from the generator
    "dropout_rate": float rate at which neurons are disabled during training
    "batchnorm": bool whether to batch-normalize after layers
    "weighted_metrics": List of metric labels to  be weighted by masking level
    "enc_dense_kwargs":Keyword arguments for encoder Dense layers
    "dec_dense_kwargs":Keyword arguments for decoder Dense layers
    "enc_node_list":List of ints representing the layers in the encoder
    "dec_node_list":List of ints representing the layers in the decoder
    "loss":String representing the loss function to use per keras labels
    "metrics":List of strings representing metrics to record per keras labels
    "max_epochs": int maximum number of epochs to train
    "train_steps_per_epoch": int number of batches per epoch
    "val_steps_per_epoch": int number of batches per validation
    "val_frequency": int epochs between validations
    "learning_rate": float learning rate of the model
    "early_stop_metric": string metric evaluated for stagnation
    "early_stop_patience": int number of epochs before stopping
    "mask_pct": float mean percentage of inputs to mask during training
    "mask_pct_stdev":stdev of number of inputs masked during training
    "mask_val": Number substituted for masked features
    "train_val_ratio": Ratio of training to validation samples during training
    "mask_feat_probs": List of relative probabilities of each feat being masked
    "notes": Optional (but recommended) string describing this model iteration

    The result is stored in the unique directory created within model_dir.
    It will contain the best-performing models as hdf5s at checkpoints,
    and the following files

    "{model-name}_config.json": serialized version of the user-defined config
    "{model-name}_summary.txt": serialized version of the user-defined config
    "{model-name}_prog.csv": keras CSVLogger output of learning curves.

    :@param X: Input data in (batch, ... , feature) format
    :@param Y: Output data in (batch, feature) format
    :@param model_config: Configuration dictionary defining the above terms.
    :@param model_dir: Parent directory where this model's dir will be created
    """
    assert X.shape[-1] == len(config["input_feats"])
    assert Y.shape[-1] == len(config["output_feats"])
    ## Initialize the model according to the configuration
    model = mm.variational_encoder_decoder(
            name=config["model_name"],
            num_inputs=len(config["input_feats"]),
            num_outputs=len(config["output_feats"]),
            latent_size=config["latent_size"],
            enc_node_list=config["enc_node_list"],
            dec_node_list=config["dec_node_list"],
            batchnorm=config["batchnorm"],
            dropout_rate=config["dropout_rate"],
            enc_dense_kwargs=config["enc_dense_kwargs"],
            dec_dense_kwargs=config["dec_dense_kwargs"],
            )

    ## Create and add some information to the dedicated model directory
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

    ## Compile the model
    model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config["learning_rate"]),
            loss=config.get("loss"),
            metrics=config.get("metrics"),
            weighted_metrics=config.get("weighted_metrics"),
            )

    ## Initialize the masking data generators
    gT,gV = mm.get_generators(
            X=X,
            Y=Y,
            tv_ratio=config["train_val_ratio"],
            noise_pct=config["mask_pct"],
            noise_stdev=config["mask_pct_stdev"],
            mask_val=config["mask_val"],
            feat_probs=config["mask_feat_probs"],
            shuffle=True,
            dtype=tf.float64
            )

    ## Define callbacks for model progress tracking
    callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=config["early_stop_metric"],
                patience=config["early_stop_patience"]
                ),
            tf.keras.callbacks.ModelCheckpoint(
                monitor="val_loss",
                save_best_only=True,
                filepath=model_dir.joinpath(
                    config['model_name']+"_{epoch:03}_{val_mae:.03f}.hdf5")
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

    ## Save the most performant models.
    ## (!!!) This relies on the checkpoint file name formatting string, (!!!)
    ## (!!!) and on that there are no other hdf5 files in the model dir (!!!)
    best_model = list(sorted(
        [q for q in model_dir.iterdir() if q.suffix==".hdf5"],
        key=lambda p:int(p.stem.split("_")[1])
        )).pop(-1)
    ## Save the model after further training without improvement
    model.save(model_dir.joinpath(config["model_name"]+"_keras.hdf5"))
    ## Save the previous best-performing model from the last checkpoint
    best_model.rename(model_dir.joinpath(config["model_name"]+"_final.hdf5"))

if __name__=="__main__":
    """ Directory with sub-directories for each model. """
    data_dir = Path("data")
    model_parent_dir = data_dir.joinpath("models")
    asos_al_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
    asos_ut_path = data_dir.joinpath("UT_ASOS_Mar_2023.csv")
    config = {
            "model_name":"ved-3",
            "input_feats":["tmpc","dwpc","relh","sknt",
                "mslp","p01m","gust","feel"],
            "output_feats":["romps_LCL_m","lcl_estimate"],
            "source_csv":asos_al_path.as_posix(),
            "batch_size":32,
            "batch_buffer":4,
            "dropout_rate":0.0,
            "batchnorm":True,
            "dense_kwargs":{"activation":"sigmoid"},
            "enc_node_list":[128,128,128,64,32],
            "latent_size":8,
            "dec_node_list":[32,64,64,16],
            "loss":"mse",
            "metrics":["mse", "mae"],
            "enc_dense_kwargs":{"activation":"relu"},
            "dec_dense_kwargs":{"activation":"relu"},
            "weighted_metrics":["mae", "mse"],
            "max_epochs":128, ## maximum number of epochs to train
            "train_steps_per_epoch":16, ## number of batches per epoch
            "val_steps_per_epoch":3, ## number of batches per validation
            "val_frequency":1, ## epochs between validation
            "learning_rate":1e-2,
            "early_stop_metric":"val_mse", ## metric evaluated for stagnation
            "early_stop_patience":30, ## number of epochs before stopping
            "mask_pct":0,
            "mask_pct_stdev":0,
            "mask_val":9999.,
            "train_val_ratio":.8,
            "mask_feat_probs":None,
            "notes":"Same architecture; NO MASKING; very fast learning rate",
            }

    from preprocess import preprocess
    data_dict = preprocess(
            asos_csv_path=asos_al_path,
            input_feats=config["input_feats"],
            output_feats=config["output_feats"],
            normalize=True,
            )

    train_variational(
            X=data_dict["X"],
            Y=data_dict["Y"],
            model_config=config,
            model_dir=model_parent_dir,
            )

