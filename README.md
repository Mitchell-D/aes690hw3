# aes690hw3

Neural networks for estimating LCL height given basic surface
meteorological data from ASOS monthly CSVs.

<p align="center">
  <img height="256" src="https://github.com/Mitchell-D/aes690hw3/blob/main/report/figs/val_ff-combined-01_combined.png" />
</p>

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

Before training, the data must be extracted and normalized.
`preprocess.py` abstracts this into a single method that returns a
dict of normalized data arrays with a last dimensions of "X" inputs
and "Y" outputs ordered to correspond with the user-provided lists of
field labels.

The models in this exercise were trained using the [tracktrain][1]
framework I've been developing to organize and generalize the process
of model fitting and evaluation. `train_one.py` implements the
workflow to train one model at a time as follows:

1. Defining a valid configuration according to `tracktrain.config`.
2. Preprocessing the data, then initializing generators with
   `tracktrain.model_methods.array_to_noisy_tv_gen`.
4. Compiling the model and creating a model directory with
   `tracktrain.ModelDir.build_from_config`.
6. Training the model with `tracktrain.compile_and_train.train`.

`train_multiple.py` extends this process to facilitate a random
search of user-defined sets of plausible hyperparameters.

Once a variety of models have been trained, `compare_models.py` is
used to specify subsets of models according to their configurations,
then to evaluate the models by calculating bulk statistics from new
datasets, and generating validation curves.


[1]:https://github.com/Mitchell-D/tracktrain
