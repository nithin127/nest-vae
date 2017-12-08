import torch.utils.data as data

class Reconstruction(data.Dataset):

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            noisy_img = self.transform(img.copy())

        if self.target_transform is not None:
            img = self.target_transform(img)

        return noisy_img, img

    def __len__(self):
        return len(self.dataset)
