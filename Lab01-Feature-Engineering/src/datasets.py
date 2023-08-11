import os, shutil
import yaml
import joblib
import pandas as pd
from .processors import Preprocessor, LUTLabelEncoder
import wget
from typing import List, Dict

# from torch import long
# from torch.utils.data import Dataset
# from transformers import T5Tokenizer


# class BaseDataset(Dataset):
class BaseDataset:
    def __init__(
        self, config_path: str, root: str = None, download: bool = False
    ) -> None:
        with open(config_path) as handler:
            config = yaml.load(handler, Loader=yaml.FullLoader)
        self.name = config["name"]
        if root is None:
            self.root = os.path.join(os.path.split(config_path)[0], self.name)
        else:
            self.root = os.path.join(root, self.name)
        self.source_url = config["source_url"]
        self.config = config

        self._parse_paths()
        if not self._validate_dataset():
            if download:
                self._download_n_extract()
            else:
                raise FileNotFoundError(
                    f"Dataset files not found in {self.root}. Use 'download=True' to download from source"
                )

    def _parse_paths(self) -> None:
        for k, v in self.config["paths"].items():
            self.config["paths"][k] = os.path.join(self.root, v)

    def _validate_dataset(self) -> bool:
        paths = self.config["paths"].values()
        paths_exist = [os.path.exists(path) for path in paths]
        valid = all(paths_exist)
        return valid

    def _download_n_extract(self) -> List[str]:
        os.makedirs(self.root, exist_ok=True)
        download_path = os.path.join(self.root, os.path.split(self.source_url)[1])
        print(f"Downloading from source ({self.source_url}) to {download_path}")
        download_path = wget.download(self.source_url, download_path)
        shutil.unpack_archive(download_path, self.root)


class IMDBDataset(BaseDataset):
    def __init__(self, config_path: str, root: str = None, download: bool = False):
        super().__init__(config_path, root, download)

        # initialize
        print("Initializing objects")
        self.preprocessor = Preprocessor()
        self.input_encoder = joblib.load(self.config["paths"]["input_encoder"])
        self.label_encoder = LUTLabelEncoder(self.config["labels"])
        # load data
        data = pd.read_csv(self.config["paths"]["data"])

        # preprocess
        print("Preprocessing")
        data.review = data.review.apply(self.preprocessor)
        self.data = data

        # encode
        x = data["review"]
        y = data["sentiment"]
        print("Encoding")
        feature_names = self.input_encoder.get_feature_names_out()
        x = self.input_encoder.transform(x)
        y = self.label_encoder.transform(y)

        # split
        ds_size = len(y)
        start_idx = 0
        end_idx = int(ds_size * self.config["split"]["train"])
        x_train = x[start_idx:end_idx]
        y_train = y[start_idx:end_idx]
        start_idx = int(ds_size * self.config["split"]["train"])
        end_idx = start_idx + int(ds_size * self.config["split"]["val"])
        x_val = x[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        start_idx = int(
            ds_size * (self.config["split"]["train"] + self.config["split"]["val"])
        )
        end_idx = -1
        x_test = x[start_idx:end_idx]
        y_test = y[start_idx:end_idx]

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.feature_names = feature_names
