import torch
from typing import Dict, Tuple, List
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset

def get_train_dataset(dataset_types: Tuple[str], train_data: Dict, tokenizer):
    train_datasets = []
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            if dataset == DataSet.__getname__():
                train_datasets.append(DataSet(**train_data, tokenizer=tokenizer))
    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")

def extend_datasets(datasets: List, dataset_items: List[str], extend: bool = False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")
                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]
                    setattr(dataset, item, value)
                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)

def create_dataloader(dataset, train_batch_size: int, shuffle: bool = False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )