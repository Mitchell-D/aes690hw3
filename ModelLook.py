import numpy as np
import json
from pathlib import Path
from itertools import chain
from collections.abc import Callable

def load_csv_prog(csv_path, as_array=False):
    """
    Load the per-epoch metrics from a tensorflow CSVLogger file as a dict.

    :@param csv_path: Path to a csv generated by tensorflow CSVLogger callback
    :@param as_array: Return a 2-tuple (labels, array) where array is a (E,M)
        shaped array for E epochs and M metrics, the metrics corresponding to
        each string label.

    :@return: a dict mapping each metric name to a list of values
    """
    csv_lines = csv_path.open("r").readlines()
    csv_lines = list(map(lambda l:l.strip().split(","), csv_lines))
    csv_labels = csv_lines.pop(0)
    csv_cols = list(map(
        lambda l:np.asarray([float(v) for v in l]),
        zip(*csv_lines)))
    if not as_array:
        return dict(zip(csv_labels, csv_cols))
    return csv_labels, np.stack(csv_cols, axis=-1)

class ModelDir:
    def __init__(self, model_dir:Path):
        self.name = model_dir.name
        self.dir = model_dir
        children = [
                f for f in model_dir.iterdir()
                if (f.name.split("_")[0]==self.name and not f.is_dir())
                ]
        self.req_files = (
                self.dir.joinpath(f"{self.name}_summary.txt"),
                self.dir.joinpath(f"{self.name}_config.json"),
                )
        self.path_summary,self.path_config = self.req_files
        self.path_prog = self.dir.joinpath(f"{self.name}_prog.csv")
        self._check_req_files()
        self._prog = None
        self._summary = None
        self._config = None

    @property
    def summary(self):
        if self._summary is None:
            self._summary = self.path_summary.open("r").read()
        return self._summary
    @property
    def prog(self):
        if self._prog is None:
            self._prog = self.load_prog(as_array=True)
        return self._prog
    @property
    def metric_labels(self):
        return self.prog[0]
    @property
    def metric_data(self):
        """
        Returns a (E,M) shaped ndarray of M metric values over E epochs.
        """
        return self.prog[1]
    @property
    def config(self):
        """ Returns the model config dictionary.  """
        if self._config == None:
            self._config = self._load_config()
        return self._config

    def get_metric(self, metric):
        """"
        Returns the per-epoch metric data array for one or more metrics.

        If a single str metric label is provided, a (E,) array of that metric's
        data over E epochs is returned.

        If a list of M str metric labels is provided, a (E,M) array of the
        corresponding metrics' data are provided (in the order of the labels).

        :@param metric: String metric label or list of metrics
        """
        if type(metric) is str:
            assert metric in self.metrics
            return self.metric_data[:,self.metric_labels.index(metric)]
        assert all(m in self.metric_labels for m in metric)
        idxs = np.array([self.metric_labels.index(m) for m in metric])
        return self.metric_data[:,idxs]

    def _check_req_files(self):
        """
        Verify that all files created by the build() function exist in the
        model directory (ie _summary.txt and _config.json).
        """
        try:
            assert all(f.exists() for f in self.req_files)
        except:
            raise FileNotFoundError(
                f"All of these files must be in {self.dir.as_posix()}:\n",
                tuple(f.name for f in self.req_files))
        return True

    def load_prog(self, as_array=False):
        """
        Load the training progress csv from a keras CSVLogger

        :@param: if True, loads progress lists as a single (E,M) ndarray
            for E epochs evaluated with M metrics
        """
        if self.path_prog is None:
            raise ValueError(
                    "Cannot return progress csv. "
                    f"File not found: {self.path_prog.as_posix()}"
                    )
        return load_csv_prog(self.path_prog, as_array=as_array)

    def _load_config(self):
        """
        Load the configuration JSON associated with a specific model as a dict.
        """
        self._config = json.load(self.path_config.open("r"))
        return self._config

    def update_config(self, update_dict:dict):
        """
        Update the config json to have the new keys, replacing any that exist.

        Overwrites and reloads the json configuration file.

        :@param update_dict: dict mapping string config field labels to new
            json-serializable values.

        :@return: the config dict after being serialized and reloaded
        """
        ## Get the current configuration and update it
        cur_config = self.config
        cur_config.update(update_dict)
        ## Overwrite the json with the new version
        json.dump(cur_config, self.path_config.open("w"))
        ## reset the config and reload the json by returning the property
        self._config = None
        return self.config

class ModelSet:
    @staticmethod
    def from_dir(model_parent_dir:Path):
        """
        Assumes every subdirectory of the provided Path is a ModelDir-style
        model directory
        """
        model_dirs = [
                ModelDir(d) for d in model_parent_dir.iterdir() if d.is_dir()
                ]
        return ModelSet(model_dirs=model_dirs)

    def __init__(self, model_dirs:list, check_valid=True):
        """ """
        ## Validate all ModelDir objects unless check_valid is False
        assert check_valid or all(m._check_req_files() for m in model_dirs)
        self._models = tuple(model_dirs)

    @property
    def models(self):
        """ return the model directories as a tuple """
        return self._models
    @property
    def model_names(self):
        """ Return the string names of all ModelDir objects in the ModelSet """
        return tuple(m.name for m in self.models)

    def subset(self, rule:Callable=None, substr:str=None, check_valid=True):
        """
        Return a subset of the ModelDir objects in this ModelSet based on
        one or both of:

        (1) A Callable taking the ModelDir object and returning True or False.
        (2) A substring that must be included in the model dir's name property.

        :@param rule: Function taking a ModelDir as the first positional arg,
            and returning True iff the ModelDir should be in the new ModelSet
        :@param substr: String that must be included in the ModelDir.name
            string property of all directories in the returned ModelSet

        :@return: ModelSet with all ModelDir objects meeting the conditions
        """
        subset = self.models
        if not rule is None:
            subset = tuple(filter(rule, subset))
        if not substr is None:
            subset = tuple(filter(lambda m:substr in m.name, subset))
        return ModelSet(subset, check_valid=check_valid)

if __name__=="__main__":
    model_parent_dir = Path("data/models")
    MS = ModelSet.from_dir(model_parent_dir)

    is_masked = lambda m:all(
            type(v) is float and v>0 for v in
            (m.config.get(l) for l in ("mask_pct", "mask_pct_stdev"))
            )
    sub = MS.subset(rule=is_masked) ## Models where masking was used
    #sub = MS.subset(substr="ved") ## Variational encoder-decoders

    print(list(sorted(m.name for m in sub.models)))
    print([m.metric_labels for m in sub.models])
    print([list(m.config.keys()) for m in sub.models])
    print([list(m.metric_data.shape) for m in sub.models])

    metric_labels,metric_data = zip(*[
        ml.load_prog(as_array=True) for ml in sub.models
        ])
    configs = [ml._load_config() for ml in sub.models]
    metric_union = set(chain(*metric_labels))
    metric_intersection = [
            m for m in metric_union
            if all(m in labels for labels in metric_labels)
            ]
    print(metric_union)
    print(metric_intersection)
