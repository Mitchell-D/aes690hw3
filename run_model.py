""" Basic script for evaluating a model trained on ASOS monthly CSV data """
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

from preprocess import preprocess

## Define the paths to the model and input CSV
asos_csv_path = Path("data/UT_ASOS_Mar_2023.csv")
model_path = Path("data/dodson_ff-alabama.keras") ## trained on AL
#model_path = Path("data/dodson_ff-combined.keras") ## trained on combined

## Extract and preprocess the data in an order determined by the features lists
data_dict = preprocess(
        asos_csv_path=asos_csv_path,
        input_feats=["tmpc", "relh", "sknt", "mslp"],
        output_feats=["romps_LCL_m", "lcl_estimate"],
        normalize=True,
        )

X = data_dict["X"]
Y = data_dict["Y"]

## Run the model and rescale the values to data coordinates
model = tf.keras.models.load_model(model_path)
P = model(X)
P = P*data_dict["y_stdevs"]+data_dict["y_means"]
Y = Y*data_dict["y_stdevs"]+data_dict["y_means"]

## Calculate the R^2 value and mean absolute error
r2_analytic,r2_estimate = tuple(r2_score(Y,P, multioutput="raw_values"))
mae_analytic,mae_estimate = tuple(np.sum(np.abs(Y-P), axis=0)/Y.shape[0])

print(f"R^2 for (Romps, 2017) prediction: {r2_analytic:.5f}")
print(f"R^2 for empirical prediction:     {r2_estimate:.5f}")
print(f"MAE for (Romps, 2017) prediction: {mae_analytic:.5f}")
print(f"MAE for empirical prediction:     {mae_estimate:.5f}")
