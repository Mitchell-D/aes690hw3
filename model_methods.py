""" Methods for building and training feedforward neural networks """
import numpy as np
import json
from pathlib import Path
from random import random

import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Layer, Dense

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
    Load the configuration JSON associated with a specific model.

    :@param model_dir: The directory associated with a specific model instance
        which contains a {model_name}_config.json describing its hyperparams.
    """
    model_name = model_dir.name
    return json.load(model_dir.joinpath(f"{model_name}_config.json").open("r"))

def load_csv_prog(model_dir):
    """
    Load the per-epoch metrics from a tensorflow CSVLogger file as a dict.

    :@param model_dir: The directory associated with a specific model instance
        which contains a {model_name}_prog.csv describing its hyperparams.
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

def kl_divergence(z_mean, z_log_var):
    """ """
    return tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            ) * -0.5

def get_vi_projection(name:str, layer_input, latent_size:int):
    """
    """
    ## Project to requested latent dimension for predicted mean and variance
    z_mean = Dense(latent_size, name=f"{name}_mean",
                   activation="linear")(layer_input)
    z_log_var = Dense(latent_size, name=f"{name}_logvar",
                      activation="linear")(layer_input)
    ## Draw a sample from the distribution from predicted parameters
    ## shaped like (B,L) for B batch elements and L latent features.
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    ## Concatenate the sample values to a (B,V,L) shape for B elements in the
    ## batch, V variational features (sample,z_mean,z_log_var), and L latent.
    return (sample,z_mean,z_log_var)

'''
class KL_Divergence(Layer):
    def call(self,inputs):
        z_sample,z_mean,z_log_var = inputs
        ## Calculate the KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
                )
        self.add_loss(kl_loss)
'''

def variational_encoder_decoder(
        name:str, num_inputs:int, num_outputs:int, latent_size:int,
        enc_node_list, dec_node_list, dropout_rate=0.0, batchnorm=True,
        enc_dense_kwargs={}, dec_dense_kwargs={}):
    """
    Construct a variational encoder decoder model parameterized by latent_size
    gaussian distributions approximating the latent posterior p(z|x) of input x

    :@param name: Model name for layer labeling
    :@param num_inputs: Number of inputs to the model
    :@param num_outputs: Number of values predicted by the model
    :@param latent_size: Number of latent distributions.
    :@param enc_node_list: List of int node counts per layer in the encoder
    :@param dec_node_list: List of int node counts per layer in the decoder
    :@param dropout_rate: Dropout rate in [0,1]
    :@param batchnorm: if True, do batchnorm regularization
    :@param enc_dense_kwargs: arguments (like activation function) passed to
        all of the encoder feedforward layers.
    :@param dec_dense_kwargs: arguments (like activation function) passed to
        all of the decoder feedforward layers.
    """
    l_input = Input(shape=(num_inputs,), name=f"{name}_input")
    l_enc_dense = get_dense_stack(
            name=f"{name}_enc",
            node_list=enc_node_list,
            layer_input=l_input,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=enc_dense_kwargs,
            )
    sample,z_mean,z_log_var = get_vi_projection(
            name=name,
            layer_input=l_enc_dense,
            latent_size=latent_size,
            )
    l_decoder = get_dense_stack(
            name=f"{name}_dec",
            node_list=dec_node_list,
            layer_input=sample,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=dec_dense_kwargs,
            )
    l_output = Dense(
            num_outputs,
            name=f"{name}_out",
            activation="linear"
            )(l_decoder)
    ved = Model(l_input, l_output)
    ved.add_loss(kl_divergence(z_mean, z_log_var))
    return ved

def feedforward(name:str, node_list:list, num_inputs:int, num_outputs:int,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """ Just a series of dense layers with some optional parameters """
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
    ved = get_ved(
            num_inputs=10,
            num_outputs=2,
            latent_size=8,
            enc_node_list=[16,32,64,64,32],
            dec_node_list=[32,64,64,32,16],
            dropout_rate=0.0,
            batchnorm=True,
            )
    ved.summary()
