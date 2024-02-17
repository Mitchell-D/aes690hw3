"""
Simple script for iterating through the model directory and collecting a list
of files that are added to the git repository, which is printed to stdout.

The model directory is assumed to only contain subdirectories that were
generated by a training script, and contain model configurations, training
metrics, and trained models; these are labeled and identified with substrings.

This script is handy for adding model info to a git repo with

git add -f `python print_model_files.py`
"""
from pathlib import Path
from itertools import chain

substrings = ["_config.json", "_summary.txt", "_final.hdf5", "_prog.csv"]
model_dir = Path("data/models")
model_files = list(chain(*[
    list(m.iterdir()) for m in model_dir.iterdir() if m.is_dir()
    ]))
matches = [f for f in model_files if any(s in f.name for s in substrings)]
print(" ".join([m.as_posix() for m in matches]))