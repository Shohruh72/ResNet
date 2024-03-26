import os
from os.path import *
from PIL import Image
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_anno(data_dir)

    def __getitem__(self, index):
        image, label = self.samples[index]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_anno(data_dir):
        assert os.path.exists(data_dir), f'{data_dir} does not exist'
        images, labels = [], []

        for root, dirs, files in os.walk(data_dir):
            label = basename(relpath(root, data_dir) if (root != data_dir) else '')
            for f in files:
                base, ext = os.path.splitext(f)
                if ext.lower() in ('.png', '.jpg', '.jpeg'):
                    images.append(os.path.join(root, f))
                    labels.append(label)

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(list(sorted(set(labels))))}
        samples = [(i, class_to_idx[j]) for i, j in zip(images, labels) if j in class_to_idx]
        return samples
