import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle as pkl

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
                    noise_stdev=mask_pct,
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
            mae += data_dict.get("y_means")
            print(f"{md.name} MAE: {[f'{v:.4f}' for v in list(mae)]}")
            ## add the MAE vector to the results dict
            results[md.name] = tuple(np.asarray(mae))
        except Exception as e:
            print(f"FAILED {md.name}")
            #print(e)
            raise e
            continue
    return results


def main():
    """ Directory with sub-directories for each model. """
    data_dir = Path("data")
    model_parent_dir = data_dir.joinpath("models")
    #asos_path,dataset_label = (data_dir.joinpath("AL_ASOS_July_2023.csv"),"AL")
    #asos_path,dataset_label = (data_dir.joinpath("UT_ASOS_Mar_2023.csv"),"UT")
    asos_path,dataset_label = (data_dir.joinpath("ASOS_combined.csv"),"combined")

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

    results = eval_models(
            model_set=sub,
            asos_csv_path=asos_path,
            rand_seed=20240128,
            dataset_label=dataset_label,
            )
    print(json.dumps(results))
    json.dump(results, data_dir.joinpath(f"mae_{dataset_label}.json").open("w"))
if __name__=="__main__":
    main()
