import os
from typing import Any

import cv2
from torch.utils.data import Dataset
import numpy as np
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
    def __init__(self, directory: str, size: tuple[int, int] = None) -> None:
        self.images_dir = os.path.join(directory, 'images')
        self.labels_dir = os.path.join(directory, 'labels')
        self.size = size
        self.images = sorted(os.listdir(self.images_dir))
        self.labels = sorted(os.listdir(self.labels_dir))

    def __len__(self) -> int:
        return len(self.images)

    @abstractmethod
    def __getitem__(self, item:int) -> tuple[np.ndarray, Any]:
        pass

    def get_image(self, idx:int) -> np.ndarray:
        image_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.size is not None:
            image = cv2.resize(image, self.size)

        return image

    def get_label(self, idx:int) -> str:
        label_path = os.path.join(self.labels_dir, self.labels[idx])
        with open(label_path, 'r') as f:
            label = f.readline().strip()
        return label


class CustomDatasetLocation(BaseDataset):
    def __init__(self, directory: str, size: tuple[int, int] = None, transform=None):
        super().__init__(directory, size)
        self.transform = transform


    def __getitem__(self, idx:int) -> tuple[np.ndarray, np.ndarray]:
        image = self.get_image(idx)
        label = self.get_label(idx)



        label = label.split(" ")[1:]
        label = np.array(list(zip(label[::2], label[1::2])), dtype=np.float32)

        if self.size is not None:
            label[:, 0] = label[:, 0].astype(np.float32) * self.size[0]
            label[:, 1] = label[:, 1].astype(np.float32) * self.size[1]

        if self.transform:
            image = self.transform(image)

        return image, label




class CustomDatasetDigit(BaseDataset):
    def __init__(self,directory: str, size: tuple[int, int] = None):
        super().__init__(directory, size)

    def get_label(self, idx:int) -> list[str]:
        label_path = os.path.join(self.labels_dir, self.labels[idx])
        with open(label_path, 'r') as f:
            label = f.readlines()
        return label

    def __getitem__(self, item) -> tuple[np.ndarray, list[list[str]]]:
        image = self.get_image(item)
        label = self.get_label(item)

        label = [l.split(" ") for l in label]
        return image, label




