import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from tracktrain.ModelDir import ModelDir,ModelSet

def scatter_2_pane(x1,x2,y1,y2,title="",xlabel="",ylabel=""):
    """ """
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.scatter(x1,y1)
    ax1.set_title()
    ax2.scatter(x2,y2)

if __name__=="__main__":
    ## dict like {model_labels:{dataset_labels:()}}
    model_parent_dir = Path("data/models")
    json_path = Path("data/eval_combined.json")
    results = json.load(json_path.open("r"))
    mlabels,run_dicts = zip(*results.items())
    mdirs = [model_parent_dir.joinpath(m) for m in mlabels]
    ms = ModelSet([ModelDir(d) for d in mdirs])
    print(ms)
