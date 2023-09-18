import os, shutil
import yaml
import joblib
import pandas as pd
import numpy as np
import wget
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from torch import Tensor
from typing import Tuple


class BaseDataset:
    def __init__(self, config_path, root=None) -> None:
        with open(config_path) as handler:
            config = yaml.load(handler, Loader=yaml.FullLoader)
        if root is None:
            self.root = os.path.abspath(os.path.split(config_path)[0])
        else:
            self.root = os.path.abspath(root)
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


class wave2vecDataset(BaseDataset):
    pass