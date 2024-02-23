import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt

import tracktrain.model_methods as mm
#from tracktrain.compile_and_train import compile_and_build_dir, train
from tracktrain.ModelDir import ModelDir,ModelSet,model_builders
from preprocess import preprocess

def eval_models(
        model_set:ModelSet, asos_csv_path, dataset_label="", rand_seed=None,
        save_pkl=True, val_ratio=1., mask_pct=0., mask_stdev=0., batch_size=64
        ):
    """
    Given a ModelSet, evaluate all of the models on the validation generator
    from a common dataset of ASOS CSV derived data.

    Only the validation generator is used so that random seeds match with the
    validation data during training. To evaluate on the full dataset, set
    val_ratio to 1.

    :@return: Dict mapping model names to vectors of MAE per output dimension.
    """
    results = {}
    for md in model_set.model_dirs:
        try:
            if md.path_final_model.exists():
                model = md.load_model(md.path_final_model)
            elif md.path_final_weights.exists():
                model_type = md.config.get("model_type")
                if model_type not in model_builders.keys():
                    print(f"Model type {model_type} ({md.name}) unrecognized")
                    continue
                model = model_builders.get(model_type)(md.config)
                model.load_weights(md.path_final_weights)
            else:
                print(f"No model found for {md}")
                continue

            ## Extract and preprocess the data
            data_dict = preprocess(
                    asos_csv_path=asos_csv_path,
                    input_feats=md.config.get("input_feats"),
                    output_feats=md.config.get("output_feats"),
                    normalize=True,
                    )
            ## Initialize the masking data generators
            gen_train,gen_val = mm.array_to_noisy_tv_gen(
                    X=data_dict["X"].astype(np.float64),
                    Y=data_dict["Y"].astype(np.float64),
                    tv_ratio=md.config.get("train_val_ratio"),
                    noise_pct=mask_pct,
                    noise_stdev=mask_stdev,
                    mask_val=md.config.get("mask_val"),
                    feat_probs=md.config.get("mask_feat_probs"),
                    shuffle=True,
                    rand_seed=rand_seed,
                    dtype=tf.float64,
                    )
            #model.summary(expand_nested=True)
            all_mae = []
            preds = []
            pkl_path = md.dir.joinpath(
                    md.name + \
                    [f"_{dataset_label}",""][dataset_label==""] + \
                    "_mae.pkl"
                    )
            for x,y,w in gen_val.batch(batch_size):
                p = tf.cast(model(x),tf.float64)
                y - tf.cast(y, tf.float64)
                all_mae.append(tf.reduce_sum(
                    tf.math.abs(p-y), axis=0)/p.shape[0])
                preds.append(p)

            if save_pkl:
                pkl.dump(np.concatenate(preds, axis=0), pkl_path.open("wb"))
            mae = tf.reduce_sum(tf.concat(all_mae, axis=0))/len(all_mae)
            mae *= data_dict.get("y_stdevs")
            print(f"{md.name} MAE: {[f'{v:.4f}' for v in list(mae)]}")
            ## add the MAE vector to the results dict
            results[md.name] = tuple(np.asarray(mae))
        except Exception as e:
            print(f"FAILED {md.name}")
            #print(e)
            raise e
            continue
    return results

def get_val_curve(model_dir:ModelDir, asos_csv_path:Path, dataset_type,
                  fig_path=None, show=True):
    """ """
    if model_dir.path_final_model.exists():
        model = model_dir.load_model(model_dir.path_final_model)
    elif model_dir.path_final_weights.exists():
        model_type = model_dir.config.get("model_type")
        if model_type not in model_builders.keys():
            raise ValueError(
                    f"Model type {model_type} ({model_dir.name}) unrecognized")
        model = model_builders.get(model_type)(model_dir.config)
        model.load_weights(model_dir.path_final_weights)
    else:
        raise ValueError(f"No model found for {model_dir}")

    ## Extract and preprocess the data
    data_dict = preprocess(
            asos_csv_path=asos_csv_path,
            input_feats=model_dir.config.get("input_feats"),
            output_feats=model_dir.config.get("output_feats"),
            normalize=True,
            )
    ## Initialize the masking data generators
    gen_train,gen_val = mm.array_to_noisy_tv_gen(
            X=data_dict["X"].astype(np.float64),
            Y=data_dict["Y"].astype(np.float64),
            tv_ratio=0,
            noise_pct=0.,
            noise_stdev=0.,
            mask_val=model_dir.config.get("mask_val"),
            feat_probs=model_dir.config.get("mask_feat_probs"),
            shuffle=True,
            rand_seed=None,
            dtype=tf.float64,
            )
    preds = []
    true = []
    for x,y,w in gen_val.batch(64):
        p = tf.cast(model(x),tf.float64)
        y = tf.cast(y, tf.float64)
        preds.append(p)
        true.append(y)
    vmax = 3500
    fig,(ax1,ax2) = plt.subplots(1,2)
    rescale = lambda d:d*data_dict["y_stdevs"]+data_dict["y_means"]
    preds = rescale(np.concatenate(preds, axis=0))
    true = rescale(np.concatenate(true, axis=0))

    '''
    res = 16
    pidxs = np.round(preds*res/vmax).astype(int)
    tidxs = np.round(true*res/vmax).astype(int)
    domain = np.zeros((res,res,pidxs.shape[0]), dtype=int)
    print(preds.shape, true.shape)
    for i in range(pidxs.shape[0]):
        for j in range(pidxs.shape[1]):
            domain[pidxs[i,j],tidxs[i,j],j] += 1
    ax1.imshow(domain[...,0])
    ax2.imshow(domain[...,1])
    '''

    ax1.scatter(true[:,0],preds[:,0], s=1)
    ax2.scatter(true[:,1],preds[:,1], s=1)
    ax1.set_ylim((0,vmax))
    ax1.set_xlim((0,vmax))
    ax2.set_ylim((0,vmax))
    ax2.set_xlim((0,vmax))

    ax1.set_xlabel(f"Analytic LCL value (m)")
    ax1.set_ylabel(f"Model prediction (m)")
    ax1.set_title(f"LCL from analytic formula")

    ax2.set_xlabel(f"Empirical LCL value (m)")
    ax2.set_title(f"LCL from empirical estimate")
    #ax2.set_ylabel(f"Model prediction")

    fig.suptitle(f"LCL predictor validation curves ({dataset_type} dataset)")
    if fig_path:
        plt.savefig(fig_path)
    if show:
        plt.show()


def main():
    """ Directory with sub-directories for each model. """
    data_dir = Path("data")
    model_parent_dir = data_dir.joinpath("models")
    asos_path,dataset_label = (data_dir.joinpath("AL_ASOS_July_2023.csv"),"AL")
    #asos_path,dataset_label = (data_dir.joinpath("UT_ASOS_Mar_2023.csv"),"UT")
    #asos_path,dataset_label = (data_dir.joinpath("ASOS_combined.csv"),"combined")

    sub = ModelSet.from_dir(model_parent_dir)

    ## Only variational autoencoders
    #sub = sub.subset(rule=lambda m:"model_type" in m.config.keys())
    ## Subset to models that take specific features
    #flabels = {"tmpc", "dwpc", "relh", "sknt", "mslp", "p01m", "gust", "feel"}
    #flabels = {'tmpc', 'relh', 'sknt', 'mslp'}
    #sub = sub.subset(
    #        rule=lambda m:set(m.config.get("input_feats")) == set(flabels))
    #sub = sub.subset(rule=lambda m:not m.config.get("model_type") is None)
    #sub = sub.subset(rule=lambda m:m.config.get("model_type") is None)
    #sub = sub.subset(rule=lambda m:m.config.get("model_type") == "ved")
    #sub = sub.subset(rule=lambda m:"rand" in m.name)

    """
    ## Routine for modifying the config file
    sub = sub.subset(rule=lambda m:m.config.get("model_type") == "ff")
    for md in sub.model_dirs:
        print(md.name, [l for l in md.config.keys() if "num_" in l])
    exit(0)
    """

    '''
    ## Routine for constructing Model objects for all of the subset's models..
    model_dirs,models = zip(*[
            (m,model_builders.get(m.config.get("model_type"))(m.config))
            for m in sub.models ])
    '''

    #'''
    sub = sub.subset(
            rule=lambda m:m.name in ("ff-combined-019", "ff-rand-014")
            )
    ## Evaluate the full ModelSet
    results = eval_models(
            model_set=sub,
            asos_csv_path=asos_path,
            rand_seed=20240128,
            dataset_label=dataset_label,
            )
    print(dataset_label)
    print(results)
    exit(0)
    #print(json.dumps(results))
    #json.dump(results, data_dir.joinpath(f"mae_{dataset_label}.json").open("w"))
    #'''

    #mname = "ff-rand-014"
    mname = "ff-combined-019"
    get_val_curve(
            ModelDir(model_parent_dir.joinpath(mname)),
            asos_csv_path=asos_path,
            dataset_type=dataset_label,
            #fig_path=Path(f"figures/val_{mname}_{dataset_label}.png"),
            show=True,
            )

if __name__=="__main__":
    main()
