# https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/make_master_file.py

# script for writing meta information of datasets into master.csv
# for graph property prediction datasets.
import pandas as pd

dataset_list = []

dataset_dict = {
    "ogbg-molbbbp": {"num tasks": 1, "eval metric": "rocauc", "download_name": "bbbp"},
    "ogbg-moltox21": {"num tasks": 12, "eval metric": "rocauc", "download_name": "tox21"},
    "ogbg-moltoxcast": {"num tasks": 617, "eval metric": "rocauc", "download_name": "toxcast"},
    "ogbg-molhiv": {"num tasks": 1, "eval metric": "rocauc", "download_name": "hiv"},
    "ogbg-molesol": {"num tasks": 1, "eval metric": "rmse", "download_name": "esol"},
    "ogbg-mollipo": {"num tasks": 1, "eval metric": "rmse", "download_name": "lipophilicity"},
}

mol_dataset_list = list(dataset_dict.keys())

for nme in mol_dataset_list:
    download_folder_name = dataset_dict[nme]["download_name"]
    dataset_dict[nme]["url"] = (
        "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/" + download_folder_name + ".zip"
    )

    if dataset_dict[nme]["eval metric"] == "rmse":
        dataset_dict[nme]["task type"] = "regression"
        dataset_dict[nme]["num classes"] = -1  # num classes is not defined for regression datasets.
    else:
        dataset_dict[nme]["task type"] = "binary classification"
        dataset_dict[nme]["num classes"] = 2


dataset_list.extend(mol_dataset_list)


OGB_DATASETS_DF = pd.DataFrame(dataset_dict)
