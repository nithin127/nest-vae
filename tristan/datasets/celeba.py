from __future__ import print_function
import torch
import torch.utils.data as data
import os
import h5py

class CelebA(data.Dataset):
    raw_folder = 'raw'
    training_file = 'celeba_64.hdf5'

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.data = h5py.File(os.path.join(self.root,
            self.raw_folder, self.training_file), 'r')
        self.features = self.data['/features']
        self.targets = self.data['/targets']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.features[index].transpose((1, 2, 0))
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.features)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
            self.raw_folder, self.training_file))

    def download(self):
        return