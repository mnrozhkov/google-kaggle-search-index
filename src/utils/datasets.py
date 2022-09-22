import json
import numpy as np
from PIL import Image

from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


class FiftyOneTorchDataset(Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        index_file_folder: Path to the folder with labels.json and
            either 'data' folder or 'manifest.json' file.
        manifest_path: Path to the manifest.json for the FiftyOne dataset.
            If left None and there is no 'manifest.json' file in 'index_file_folder',
            then it is assumed that images 'index_file_folder/data'.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset 
            that contains the desired labels to load
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    def __init__(
        self,
        index_file_folder,
        manifest_path=None,
        transform=None,
        gt_field="ground_truth",
        target_transform=None,
    ):
        self.index_file_folder = Path(index_file_folder)
        self.labels_path = self.index_file_folder/'labels.json'
        self.manifest_path = manifest_path
        self.transform = transform
        self.gt_field = gt_field
        self.target_transform = target_transform

        with open(self.labels_path) as json_file:
            self.labels = json.load(json_file)

        if (index_file_folder/'manifest.json').is_file():
            self.manifest_path = index_file_folder/'manifest.json'

        if self.manifest_path is None:
            files = [p for p in (index_file_folder/'data').iterdir() if p.is_file()]
            self.manifest = {file.name.split('.')[0]:file.resolve().as_posix() for file in files}
        else:
            with open(self.manifest_path) as json_file:
                self.manifest = json.load(json_file)

        self.img_paths = [self.manifest[img_name] for img_name in self.labels['labels'].keys()]

        self.classes = np.unique(list(self.labels['labels'].values())).tolist()

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[lbl] for lbl in self.labels['labels'].values()]
        self.samples = [(self.manifest[img_name],self.class_to_idx[lbl]) for img_name, lbl in self.labels['labels'].items()]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_classes(self) -> Tuple[Any]:
        return self.classes