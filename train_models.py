import numpy as np
import tensorflow as tf
from pathlib import Path

import tracktrain.model_methods as mm
from tracktrain.compile_and_train import compile_and_build_dir, train
from preprocess import preprocess
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder

def main():
    """ Directory with sub-directories for each model. """
    data_dir = Path("data")
    model_parent_dir = data_dir.joinpath("models")
    asos_al_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
    asos_ut_path = data_dir.joinpath("UT_ASOS_Mar_2023.csv")

    #input_feats = ["tmpc","dwpc","relh","sknt", "mslp","p01m","gust","feel"]
    input_feats = ["tmpc","relh","sknt","mslp"]
    output_feats = ["romps_LCL_m","lcl_estimate"]

    model_builders = {
            "vae":VariationalEncoderDecoder,
            "ff":mm.feedforward,
            }

    config = {
            ## Meta-info
            "model_name":"test-7",
            "num_inputs":len(input_feats),
            "num_outputs":len(output_feats),
            "data_source":asos_al_path.as_posix(),
            "input_feats":input_feats,
            "output_feats":output_feats,
            "model_type":"vae",

            ## Exclusive to feedforward
            "node_list":[64,32,32,32,16,16],
            "dense_kwargs":{"activation":"sigmoid"},

            ## Exclusive to variational encoder-decoder
            "num_latent":6,
            "enc_node_list":[128,128,128,64,32],
            "dec_node_list":[6],
            "dropout_rate":0.0,
            "batchnorm":True,
            "enc_dense_kwargs":{"activation":"relu"},
            "dec_dense_kwargs":{"activation":"relu"},

            ## Common to models
            "batchnorm":True,
            "dropout_rate":0.1,

            ## Exclusive to compile_and_build_dir
            "learning_rate":1e-5,
            "loss":"mse",
            "metrics":["mse", "mae"],
            "weighted_metrics":["mse", "mae"],

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
            "mask_pct":0.0,
            "mask_pct_stdev":0.2,
            "mask_val":0,
            "mask_feat_probs":None,

            "notes":"No dropout, single-layer decoder, larger latent vector",
            }

    """
    --( Exclusive to generator init for testing )--

    "train_val_ratio": Ratio of training to validation samples during training
    "mask_pct": Float mean percentage of inputs to mask during training
    "mask_pct_stdev": Stdev of number of inputs masked during training
    "mask_val": Number substituted for masked features
    "mask_feat_probs": List of relative probabilities of each feat being masked
    """
    ## Preprocess the data
    data_dict = preprocess(
            asos_csv_path=asos_al_path,
            input_feats=input_feats,
            output_feats=output_feats,
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
    model = model_builders.get(config.get("model_type"))(**config)

    """ All the methods below from this module are model and data agnostic """

    ## Compile the model and build a directory for it
    model,model_dir = compile_and_build_dir(
            model=model,
            model_parent_dir=model_parent_dir,
            compile_config=config,
            )
    best_model = train(
            model_dir=model_dir,
            train_config=config,
            compiled_model=model,
            gen_training=gen_train,
            gen_validation=gen_val,
            )
    print(f"Best model: {best_model.as_posix()}")

if __name__=="__main__":
    main()
