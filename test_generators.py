import tensorflow as tf
from pathlib import Path
import numpy as np

from train_feedforward import get_generators,gen_noisy
from preprocess import preprocess

if __name__=="__main__":
    data_dir = Path("data")
    asos_al_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
    data_dict = preprocess(
            asos_csv_path=asos_al_path,
            input_feats=["tmpc","dwpc","relh","sknt","mslp","p01m","gust"],
            output_feats=["romps_LCL_m","lcl_estimate"],
            normalize=True,
            )
    g = gen_noisy(
            X=data_dict["X"],
            Y=data_dict["Y"],
            noise_pct=0,
            noise_stdev=.4,
            mask_val=9999.,
            feat_probs=None,
            shuffle=True,
            )
    for i in range(10):
        x,y,w = next(g)
        print([f"{v:.2f}" for v in list(x)], w)
    '''
    gT,gV = get_generators(
            X=data_dict["X"],
            Y=data_dict["Y"],
            tv_ratio=.8,
            noise_pct=.2,
            noise_stdev=.4,
            mask_val=9999.,
            feat_probs=None,
            shuffle=True,
            dtype=tf.float64
            )
    for x,y,w in gT.batch(6):
        print(x.numpy())
        print([f"{v:.2f}" for v in list(np.squeeze(x.numpy()))])
        print([f"{v:.2f}" for v in list(np.squeeze(y.numpy()))])
        print(f"{w.numpy():.2f}")

    '''

    #print()
    #for v in gV.batch(1):
    #    print(v)
