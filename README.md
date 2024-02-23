# aes690hw3

Neural networks for estimating LCL height given basic surface
meteorological data from ASOS monthly CSVs.

## dependencies

 - tracktrain
 - tensorflow>=2.14
 - sklearn
 - numpy

## quickly evaluating models

The best models trained on Alabama-only data and Alabama data
combined with Utah data are saved at `data/dodson_ff-alabama.keras`
and `data/dodson_ff-combined.keras`, respectively. In order to
quickly load and evaluate bulk statistics on new data, clone this
repository and change the `asos_csv_path` variable in `run_model.py`
to point to a new file, then run the script.

The model takes a (B,4) shaped array as an input for B samples of the
4 features ["tmpc", "relh", "sknt", "mslp"], and generates a (B,2)
shaped array assigning each of the B samples to predicted values for
romps\_lcl\_m and lcl\_estimate.

To evaluate the model, execute `run_model.py` after updating the
variable `asos_csv_path` with the new monthly data file. The script
will use `preprocess.py` to prepare the data, execute the model,
and report R^2 and mean absolute error metrics for the predictions.

## training models

The models in this exercise were trained using the [tracktrain][1]
framework I've been developing to organize and generalize the process
of model fitting and evaluation. `train_one.py`

[1]:https://github.com/Mitchell-D/tracktrain
