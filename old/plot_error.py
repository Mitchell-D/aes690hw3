import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from tracktrain.ModelDir import ModelDir,ModelSet

def scatter(ds_label, models_dict):
    """ """
    fig,ax = plt.subplots()
    ## Dots: trained on combined AL+UT; Cross: trained on only AL
    comb = {k:v for k,v in d.items() if "combined" in k}
    al_only = {k:v for k,v in d.items() if "combined" not in k}


    mae_romps, mae_est = zip(*comb.values())
    mnames,maes = zip(*comb.items())
    mae_romps,mae_est = zip(*maes)
    ## Blue: feed-forward, Red: variational encoder-decoder
    colors = [("r","b")["ff" in l] for l in mnames]
    ax.scatter(mae_romps, mae_est, c=colors, marker="o")

    mnames,maes = zip(*al_only.items())
    mae_romps,mae_est = zip(*maes)
    ## Blue: feed-forward, Red: variational encoder-decoder
    colors = [("r","b")["ff" in l] for l in mnames]
    mae_romps, mae_est = zip(*al_only.values())
    ax.scatter(mae_romps, mae_est, c=colors, marker="x")

    ax.set_title(f"Models evaluated on {k}")
    ax.set_xlabel("MAE wrt LCL analytic calculation")
    ax.set_xlabel("MAE wrt LCL empirical estimate")
    plt.xlim((0,2500))
    plt.ylim((0,2500))
    plt.show()

if __name__=="__main__":
    ## dict like {model_labels:{dataset_labels:()}}
    model_parent_dir = Path("data/models")
    datasets = {
            d:json.load(Path(f"data/mae_{d}.json").open("r"))
            for d in ["combined", "UT", "AL"]
            }

    #for k,d in datasets.items():
    #    scatter(k.d)

