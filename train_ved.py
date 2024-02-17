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

:@param X: Input data in (batch, ... , feature) format.
:@param Y: Output data in (batch, feature) format
:@param model_config: Configuration dictionary defining the above terms.
:@param model_dir: Parent directory where this model's dir will be created
"""
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
from typing import Iterator

import model_methods as mm
from preprocess import preprocess
from VED import VED

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

def validate_keys(mandatory_keys:list, received_keys:list, source_name:str="",
                  descriptions:dict={}):
    """
    Raises a descriptive ValueError if a subset of the provided list of keys
    aren't in recieved_keys.

    :@param mandatory_keys: list of keys that must appear in received_keys
    :@param received_keys:  list of keys that were provided by the user
    :@param source_name: name of pipeline configured by the keys
        (ie "train", "compile")
    :@param descriptions: dict mapping at least some of the keys to
        descriptive strings to guide the user in providing an appropriate
        parameter as an argument.
    """
    ## (bool, str) 2-tuple of mandatory args and whether they are provided
    args = [(a in received_keys, a) for a in mandatory_keys]
    excluded_args = list(filter(lambda a:not a[0], args))
    if len(excluded_args)>0:
        ## Add a space if a source name was provided
        if source_name != "":
            source_name += " "
        err_str = f"{source_name}config missing mandatory fields:\n"
        missing_fields = tuple(zip(*excluded_args))[1]
        ## If descriptions are provided, print an error message including the
        ## description of each missing parameter having one.
        if not descriptions:
            err_str += ", ".join(missing_fields)
        else:
            print(missing_fields)
            descs = [(f,f"{f}:{descriptions.get(f)}")[f in descriptions.keys()]
                     for f in missing_fields]
            err_str += "\n    ".join(descs)
        raise ValueError(err_str)
    return True

def train(model_dir, train_config:dict, compiled_model:Model,
          gen_training:Iterator, gen_validation:Iterator):
    """
    Execute the training pipeline

    (1) Ensure all mandatory arguments are provided or have defaults
    (2) Initialize the callbacks
    (3) Fit the model using the data generators and configuration values
    (4) Save the model the best minimizes the

    :@param model_dir: Existing model_dir with a _summary.txt and _config.json
    :@param train_config:
    """
    mandatory_args = (
            "model_name", "early_stop_metric", "early_stop_patience",
            "save_weights_only", "batch_size", "batch_buffer", "max_epochs",
            "val_freq"
            )
    train_arg_descriptions = {
            "model_name": \
                    "Unique string name for this model, which must match the "
                    "name of the model directory",
            "early_stop_metric":"\
                    string metric evaluated during training to track "
                    "learning stagnation and determine when to stop training",
            "early_stop_patience": \
                    "int number of epochs before stopping",
            "save_weights_only": \
                    "If True, ModelCheckpoint will only save model weights "
                    "(as .weights.hdf5 files), instead of the full model "
                    "metadata. This should be used if a custom class or loss "
                    "function can't serialize",
            "batch_size": \
                    "int minibatch size (samples per weight update)",
            "batch_buffer": \
                    "int num of generator batches to preload in memory",
            "max_epochs": \
                    "int maximum number of epochs to train",
            "val_freq": \
                    "int epochs between validations",
            }
    train_arg_defaults = {
            "callbacks":["early_stop","model_checkpoint","csv_logger"],
            "early_stop_metric":"val_loss",
            "early_stop_patience":20,
            "save_weights_only":False,
            "batch_size":32,
            "batch_buffer":3,
            "max_epochs":512,
            "val_freq":1,
            }
    assert model_dir.exists()
    assert model_dir.name == train_config.get("model_name")
    ## train_config has order precedence in the dict re-composition
    train_config = {**train_arg_defaults, **train_config}
    ## Make sure the mandatory keys (or defaults) are in the dictionary
    validate_keys(
            mandatory_keys=mandatory_args,
            received_keys=list(train_config.keys()),
            source_name="train",
            descriptions=train_arg_descriptions,
            )
    ## Choose the model save file path based on whether only weights are stored
    halt_metric = train_config.get("early_stop_metric")
    if train_config.get("save_weights_only"):
        out_path = model_dir.joinpath(
                train_config.get('model_name') + \
                        "_{epoch:03}_{val_loss:.03f}.weights.hdf5"
                        )
    else:
        out_path = model_dir.joinpath(
                train_config.get("model_name") + \
                        "_{epoch:03}_{val_loss:.03f}.hdf5",
                        )
    callbacks = {
            "early_stop":tf.keras.callbacks.EarlyStopping(
                monitor=halt_metric,
                patience=train_config.get("early_stop_patience")
                ),
            "model_checkpoint":tf.keras.callbacks.ModelCheckpoint(
                monitor=halt_metric,
                save_best_only=True,
                filepath=out_path.as_posix(),
                save_weights_only=train_config.get("save_weights_only"),
                ),
            "csv_logger":tf.keras.callbacks.CSVLogger(
                model_dir.joinpath(f"{train_config.get('model_name')}_prog.csv"),
                )
            }
    for c in train_config.get("callbacks"):
        c = [callbacks[c] if c in callbacks.keys() else c]

    ## Train the model on the generated tensors
    hist = compiled_model.fit(
            gen_training.batch(train_config.get("batch_size")).prefetch(
                train_config.get("batch_buffer")),
            epochs=train_config.get("max_epochs"),
            ## Number of batches to draw per epoch. Use full dataset by default
            #steps_per_epoch=train_config.get("train_steps_per_epoch"),
            validation_data=gen_validation.batch(
                train_config.get("batch_size")
                ).prefetch(train_config.get("batch_buffer")),
            ## batches of validation data to draw per epoch
            #validation_steps=train_config.get("val_steps_per_epoch"),
            ## Number of epochs to wait between validation runs.
            validation_freq=train_config.get("val_frequency"),
            callbacks=callbacks,
            verbose=2,
            )

    ## Save the most performant model from the last checkpoint
    ## (!!!) This relies on the checkpoint file name formatting string, (!!!)
    ## (!!!) and on that there are no non-model .hdf5 files in the dir. (!!!)
    best_model = list(sorted(
        [q for q in model_dir.iterdir() if ".hdf5" in q.suffix],
        key=lambda p:int(p.stem.split("_")[1])
        )).pop(-1)
    ## Call the suffix so Path is {model_name}_final( .hdf5 | .weights.hdf5 )
    ## conditional on if the full Model object or just the weights are stored.
    best_model.rename(model_dir.joinpath(
        train_config.get("model_name")+"_final"+"".join(
            [s for s in best_model.suffixes if s in (".weights",".hdf5",".h5")]
            )))
    return best_model

def compile_and_build_dir(
        model, model_parent_dir:Path, build_config:dict, print_summary=True
        ):
    """
    Run the model build pipeline, which makes sure that the model can compile


    (1) Create a model directory with a _config.json
    (2) Compile the model with Adam and the requested metrics
    (3) Write a _summary.txt file to the model directory, implicitly verifying
        that the model can output a summary given the configuration.

    :@return: 2-tuple (model, model_dir_path)
    """
    mandatory_args = (
            "model_name", "input_feats", "output_feats", "latent_size",
            "enc_node_list", "dec_node_list", "dropout_rate", "batchnorm",
            "enc_dense_kwargs", "dec_dense_kwargs", "learning_rate", "metrics",
            "weighted_metrics"
            )
    build_arg_descriptions = {
            "learning_rate":"Learning rate ceiling for Adam optimizer",
            "metrics":"List of Metric objects or metric names to track ",
            "weighted_metrics":"Metrics to scale by generated sample weights",
            "loss":"loss function to use for training"
            }
    build_arg_defaults = {
            "weighted_metrics":None,
            "loss":"mse"
            }
    ## build_config has order precedence in the dict re-composition
    build_config = {**build_arg_defaults, **config}

    validate_keys(
            mandatory_keys=mandatory_args,
            received_keys=list(build_config.keys()),
            source_name="build",
            descriptions=build_arg_descriptions,
            )

    ## Compile the model
    model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=build_config.get("learning_rate")),
            metrics=build_config.get("metrics"),
            loss=build_config.get("loss"),
            weighted_metrics=build_config.get("weighted_metrics"),
            )

    ## Once we know that the model can compile, create and add some
    ## information to the dedicated model directory
    model_dir = model_parent_dir.joinpath(build_config.get("model_name"))
    assert not model_dir.exists()
    model_dir.mkdir()
    model_json_path = model_dir.joinpath(
            f"{build_config.get('model_name')}_config.json")
    model_json_path.open("w").write(json.dumps(build_config,indent=4))

    ## Write a summary of the model to a file
    if print_summary:
        ## Write a model summary to stdout and to a file
        model.summary()
    summary_path = model_dir.joinpath(
            build_config.get("model_name")+"_summary.txt")
    with summary_path.open("w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    ## Return the compiled model
    return model,model_dir

def get_ved(ved_config:dict):
    """
    Returns an uncompiled variational encoder-decoder Model if the provided
    configuration dictionary is sufficient for doing so.
    """
    mandatory_args = (
            "model_name", "input_feats", "output_feats", "latent_size",
            "enc_node_list", "dec_node_list", "dropout_rate", "batchnorm",
            "enc_dense_kwargs", "dec_dense_kwargs"
            )
    ved_arg_descriptions = {
            ## no defaults
            "model_name": \
                    "Unique strign name of this model, which determines"
                    "the name of the model directory ",
            "input_feats": \
                    "List of unique strings labeling the data on the final "
                    "axis of the input tensor",
            "output_feats": \
                    "List of unique strings labeling the data on the final "
                    "axis of the output tensor",
            "latent_size":"Int dimensionality of the latent distribution",
            "enc_node_list":"List of integer layer widths for the encoder",
            "dec_node_list":"List of integer layer widths for the decoder",
            ## have defaults
            "dropout_rate":"Float dropout rate in Dense layers",
            "batchnorm":"Boolean on whether to use batchnorm in Dense layers",
            "enc_dense_kwargs":"Keyword args to pass to encoder Dense layers",
            "dec_dense_kwargs":"Keyword args to pass to decoder Dense layers",
            }
    ved_arg_defaults = {
            "dropout_rate":0.0,
            "batchnorm":True,
            "enc_dense_kwargs":{},
            "dec_dense_kwargs":{},
            }
    validate_keys(
            mandatory_keys=mandatory_args,
            received_keys=list(ved_config.keys()),
            source_name="get_ved",
            descriptions=ved_arg_defaults,
            )
    ## config has order precedence in the dict re-composition
    ved_config = {**ved_arg_defaults, **ved_config}

    return VED(
            model_name=ved_config.get("model_name"),
            num_inputs=len(ved_config.get("input_feats")),
            num_outputs=len(ved_config.get("output_feats")),
            num_latent=ved_config.get("latent_size"),
            enc_node_list=ved_config.get("enc_node_list"),
            dec_node_list=ved_config.get("dec_node_list"),
            dropout_rate=ved_config.get("dropout_rate"),
            batchnorm=ved_config.get("batchnorm"),
            enc_dense_kwargs=ved_config.get("enc_dense_kwargs"),
            dec_dense_kwargs=ved_config.get("dec_dense_kwargs"),
            )



if __name__=="__main__":
    """ Directory with sub-directories for each model. """
    data_dir = Path("data")
    model_parent_dir = data_dir.joinpath("models")
    asos_al_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
    asos_ut_path = data_dir.joinpath("UT_ASOS_Mar_2023.csv")
    config = {
            ## Meta-info
            "model_name":"ved-7",
            "input_feats":["tmpc","relh","sknt","mslp"],
            #"input_feats":["tmpc","dwpc","relh","sknt",
            #    "mslp","p01m","gust","feel"],
            "output_feats":["romps_LCL_m","lcl_estimate"],
            "source_csv":asos_al_path.as_posix(),


            ## Exclusive to get_ved
            "latent_size":8,
            "enc_node_list":[128,128,128,64,32],
            "dec_node_list":[16,16],
            "dropout_rate":0.1,
            "batchnorm":True,
            "enc_dense_kwargs":{"activation":"relu"},
            "dec_dense_kwargs":{"activation":"relu"},

            ## Exclusive to compile_and_build_dir
            "learning_rate":1e-6,
            "metrics":["mse", "mae"],
            "weighted_metrics":["mae", "mse"],

            ## Exclusive to train
            "early_stop_metric":"val_mse", ## metric evaluated for stagnation
            "early_stop_patience":30, ## number of epochs before stopping
            "save_weights_only":True,
            "batch_size":32,
            "batch_buffer":4,
            "max_epochs":128, ## maximum number of epochs to train
            "val_frequency":1, ## epochs between validation

            ## Exclusive to generator init
            "train_val_ratio":.8,
            "mask_pct":0.,
            "mask_pct_stdev":0.0,
            "mask_val":999.,
            "mask_feat_probs":None,

            "notes":"New framework; slow learning rate; small decoder; some dropout",
            }

    ## Preprocess the data
    data_dict = preprocess(
            asos_csv_path=asos_al_path,
            input_feats=config.get("input_feats"),
            output_feats=config.get("output_feats"),
            normalize=True,
            )

    ## Initialize the masking data generators
    gen_train,gen_val = mm.array_to_noisy_tv_gen(
            X=data_dict["X"].astype(np.float64),
            Y=data_dict["Y"].astype(np.float64),
            tv_ratio=config.get("train_val_ratio"),
            noise_pct=config.get("mask_pct"),
            noise_stdev=config.get("mask_pct_stdev"),
            mask_val=config.get("mask_val"),
            feat_probs=config.get("mask_feat_probs"),
            shuffle=True,
            dtype=tf.float64
            )

    ## Initialize the model
    model = get_ved(
            ved_config=config
            )

    ## Compile the model and build a directory for it
    model,model_dir = compile_and_build_dir(
            model=model,
            model_parent_dir=model_parent_dir,
            build_config=config,
            )
    best_model = train(
            model_dir=model_dir,
            train_config=config,
            compiled_model=model,
            gen_training=gen_train,
            gen_validation=gen_val,
            )
    print(f"Best model: {best_model.as_posix()}")
