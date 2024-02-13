
import numpy as np
from datetime import datetime
from pathlib import Path

def parse_csv(csv_path:Path, fields:list=None, replace_val=np.nan):
    """
    Parse a monthly ASOS report for a user specified number/ordering of fields,
    converting to datetime or float value where appropriate, and replacing
    missing float values with the requested number.

    :@param csv_path: valid path to ASOS style csv
    :@param fields: Optionally specify a subset of the fields in parsing order.
    :@param replace_val: Value to replace null entries in float fields.
    """
    str_fields = ["station", "skyc1", "skyl1", "sknt"]
    time_fields = ["valid"]
    all_lines = csv_path.open("r").readlines()
    labels = all_lines.pop(0).strip().split(",")
    all_cols = list(zip(*map(lambda l:l.strip().split(","),all_lines)))
    if fields is None:
        fields = labels
    else:
        assert all(f in labels for f in fields)

    data = []
    for f in fields:
        idx = labels.index(f)
        if labels[idx] in time_fields:
            data.append(tuple(map(
                lambda t:datetime.strptime(t,"%Y-%m-%d %H:%M"),
                all_cols[idx]
                )))
        elif labels[idx] not in str_fields:
            data.append(tuple(map(float,
                ([v,replace_val][v==""] for v in all_cols[idx])
                )))
        else:
            data.append(tuple(all_cols[idx]))
    return fields,data

if __name__=="__main__":
    data_dir = Path("data")
    csv_path = data_dir.joinpath("AL_ASOS_July_2023.csv")

    labels,data = parse_csv(
            csv_path=csv_path,
            fields=["station", "valid", "tmpc", "dwpc", "relh", "sknt",
                    "mslp", "p01m", "gust", "romps_LCL_m", "lcl_estimate"],
            replace_val=0,
            )
    for d in data:
        print(d[::100])
