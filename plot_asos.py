
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from preprocess import preprocess
from krttdkit.operate import enhance as enh

def main():
    data_dir = Path("data")
    ## Preprocess the monthly data from ASOS stations
    data_dict = preprocess(
            asos_csv_path=data_dir.joinpath("AL_ASOS_July_2023.csv"),
            input_feats=["tmpc","dwpc","relh","sknt","mslp","p01m","gust","feel"],
            output_feats=["romps_LCL_m","lcl_estimate"],
            normalize=False,
            )

    print(f"Preprocessed data keys: {list(data_dict.keys())}")

    ## Collect all the data together along the feature axis
    station_labels = list(set(data_dict["stations"]))
    all_data = np.concatenate((data_dict["X"], data_dict["Y"]), axis=-1)
    feat_labels = data_dict["x_labels"] + data_dict["y_labels"]

    ## Add times of day as a feature
    times_of_day = np.array([
        float(t.strftime("%H"))*60+float(t.strftime("%M"))
        for t in data_dict["times"]])
    feat_labels.append("tod")
    all_data = np.concatenate((all_data, times_of_day[:,None]), axis=-1)
    print(all_data.shape, len(feat_labels))

    ## get a list of 2-tuples (station_label, data) for each station
    station_idxs = np.array([
        station_labels.index(s) for s in data_dict["stations"]
        ])
    station_data = {
            station_labels[i]:all_data[station_idxs==i]
            for i in range(len(station_labels))
            }

    ## Get histogram information using a method from krttdkit.operate.enhance
    ## https://github.com/Mitchell-D/krttdkit/blob/main/krttdkit/operate
    print(feat_labels)
    #feat = "gust"
    #feat = "tmpc"
    plot_feats = ["romps_LCL_m", "lcl_estimate"]
    #plot_feats = ["tmpc"]
    fig,ax = plt.subplots()
    #'''
    for i in range(len(station_labels)):
        slabel = station_labels[i]
        feat_hists = {
                feat_labels[j]:enh.do_histogram_analysis(
                    station_data[station_labels[i]][...,j],nbins=16)
                for j in [feat_labels.index(f) for f in plot_feats]}
        for f in plot_feats:
            ax.plot(feat_hists[f]["domain"],feat_hists[f]["hist"],label=slabel)
    ax.legend()
    plt.show()
    #'''
    #'''
    fig,ax = plt.subplots()
    full_hist = {feat_labels[i]: \
                 enh.do_histogram_analysis(all_data[...,i], nbins=24)
                 for i in range(len(feat_labels))}
    for f in plot_feats:
        ax.plot(full_hist[f]["domain"], full_hist[f]["hist"],
                label=feat_labels[feat_labels.index(f)])
    ax.legend()
    plt.show()
    #'''

if __name__=="__main__":
    main()
