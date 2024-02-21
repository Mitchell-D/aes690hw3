"""
Script for randomly searching a user-defined combinatorial graph
of model configurations within the tracktrain framework.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

import tracktrain.model_methods as mm
#from tracktrain.compile_and_train import compile_and_build_dir, train
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir
from tracktrain.compile_and_train import train

data_dir = Path("data")
asos_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
model_parent_dir = data_dir.joinpath("models")
#asos_path = data_dir.joinpath("UT_ASOS_Mar_2023.csv")

#input_feats = ["tmpc","dwpc","relh","sknt", "mslp","p01m","gust","feel"]
input_feats = ["tmpc","relh","sknt","mslp"]
output_feats = ["romps_LCL_m","lcl_estimate"]


"""
base_config contains configuration values to all models,
so it should only have fields not subject to variations.
"""
base_config = {
        ## Meta-info
        #"model_name":"test-15",
        "num_inputs":len(input_feats),
        "num_outputs":len(output_feats),
        "data_source":asos_path.as_posix(),
        "input_feats":input_feats,
        "output_feats":output_feats,
        "model_type":"ff",
        "rand_seed":20240128,

        ## Exclusive to feedforward
        #"node_list":[64,64,32,32,16],
        #"dense_kwargs":{"activation":"relu"},

        ## Exclusive to variational encoder-decoder
        #"num_latent":8,
        #"enc_node_list":[64,64,32,32,16],
        #"dec_node_list":[16,32,32,64],
        #"enc_dense_kwargs":{"activation":"relu"},
        #"dec_dense_kwargs":{"activation":"relu"},

        ## Common to models
        "batchnorm":True,
        #"dropout_rate":0.0,

        ## Exclusive to compile_and_build_dir
        #"learning_rate":1e-5,
        "loss":"mse",
        "metrics":["mse", "mae"],
        "weighted_metrics":["mse", "mae"],

        ## Exclusive to train
        "early_stop_metric":"val_mse", ## metric evaluated for stagnation
        "early_stop_patience":64, ## number of epochs before stopping
        "save_weights_only":True,
        "batch_size":64,
        "batch_buffer":4,
        "max_epochs":2048, ## maximum number of epochs to train
        "val_frequency":1, ## epochs between validation

        ## Exclusive to generator init
        "train_val_ratio":.9,
        "mask_pct":0.0,
        "mask_pct_stdev":0.0,
        "mask_val":9999,
        "mask_feat_probs":None,

        "notes":"",
        }

"""
The variations dictionary maps config field names to tuples containing valid
values for that field. For each new model, one of the configurations in
the tuple corresponding to each field will be selected as that field's
parameter for the model run.

If many variations are specified, the combinatorial space will be too large
to fully search. In order to evaluate a wide variety of possible hyper-
-parameterizations, a random selection is made for each field at every run.
"""

num_samples = 32
model_base_name = "ff-rand"
variations = {
        "dropout_rate":(0.0,0.1,0.2,0.4),
        "learning_rate":(1e-6,1e-4,1e-2),
        "train_val_ratio":(.6,.8,.9),
        "mask_pct":(0,0,0,.1,.2,.3),
        "mask_pct_stdev":(0,0,0,.1,.2),

        ## FF only
        "node_list":(
            (32,32,16),
            (64,64,32,32,16),
            (64,64,64,32,32,32,16),
            (256,256,256,64,64,64,32,32,32,16),
            (16,32,32,32,64,64,64,256,256,256),
            (16,32,32,32,64,64,64),
            (16,32,32,64,64),
            (16,32,32),
            ),
        "dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            {"activation":"tanh"},
            ),

        ## VED only
        "num_latent":(4,8,12),
        "enc_node_list":(
            (32,32,16),
            (64,64,32,32,16),
            (64,64,64,32,32,32,16),
            (256,256,256,64,64,64,32,32,32,16),
            ),
        "dec_node_list":(
            (16,32,32),
            (16,32,32,64,64),
            (16,32,32,32,64,64,64),
            (16,32,32,32,64,64,64,256,256,256),
            ),
        "enc_dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            {"activation":"tanh"},
            ),
        "dec_dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            {"activation":"tanh"},
            ),
        }

vlabels,vdata = zip(*variations.items())

comb_failed = []
comb_trained = []
vdata = tuple(map(tuple, vdata))
comb_shape = tuple(len(v) for v in vdata)
comb_count = np.prod(np.array(comb_shape))
for i in range(num_samples):
    ## Get a random argument combination from the configuration
    cur_comb = tuple(np.random.randint(0,j) for j in comb_shape)
    cur_update = {
            vlabels[i]:vdata[i][cur_comb[i]]
            for i in range(len(vlabels))
            }
    cur_update["model_name"] = model_base_name+f"-{i:03}"
    cur_config = {**base_config, **cur_update}
    try:
        ## Build a config dict for the selected current combination

        ## Extract and preprocess the data
        from preprocess import preprocess
        data_dict = preprocess(
                asos_csv_path=asos_path,
                input_feats=input_feats,
                output_feats=output_feats,
                normalize=True,
                )
        ## Initialize the masking data generators
        gen_train,gen_val = mm.array_to_noisy_tv_gen(
                X=data_dict["X"].astype(np.float64),
                Y=data_dict["Y"].astype(np.float64),
                tv_ratio=cur_config.get("train_val_ratio"),
                noise_pct=cur_config.get("mask_pct"),
                noise_stdev=cur_config.get("mask_pct_stdev"),
                mask_val=cur_config.get("mask_val"),
                feat_probs=cur_config.get("mask_feat_probs"),
                shuffle=True,
                dtype=tf.float64,
                rand_seed=cur_config.get("random_seed"),
                )
        ## Initialize the model
        model,md = ModelDir.build_from_config(
                cur_config,
                model_parent_dir=model_parent_dir,
                print_summary=False,
                )
        best_model = train(
            model_dir_path=md.dir,
            train_config=cur_config,
            compiled_model=model,
            gen_training=gen_train,
            gen_validation=gen_val,
            )
    except Exception as e:
        print(f"FAILED update combination {cur_update}")
        print(e)
        comb_failed.append(cur_comb)
    comb_trained.append(cur_comb)
