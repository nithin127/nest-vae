from __future__ import print_function
import torch
import torch.utils.data as data
from PIL import Image
import os
import errno

class DSprites(data.Dataset):
    url = ('https://github.com/deepmind/dsprites-dataset/raw/master/'
           'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.train_data, self.train_labels = torch.load(
            os.path.join(self.root, self.processed_folder, self.training_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
            self.processed_folder, self.training_file))

    def download(self):
        from six.moves import urllib
        import numpy as np

        if self._check_exists():
            return

        # Download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        filename = os.path.basename(self.url)
        file_path = os.path.join(self.root, self.raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        # Process and save as torch files
        print('Processing...')

        with open(file_path, 'rb') as f:
            raw_dataset = np.load(f)
            training_set = (
                torch.ByteTensor(raw_dataset['imgs']),
                torch.ByteTensor(raw_dataset['latents_classes'])
            )

        with open(os.path.join(self.root, self.processed_folder,
                self.training_file), 'wb') as f:
            torch.save(training_set, f)

        print('Done!')
