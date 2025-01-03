import torch
from torch.utils import data
import numpy as np
import imageio
import torch.nn.functional as F
import random


class Dataset2D(data.Dataset):
    def __init__(self, image_paths, target_paths, transform=None, transform_label=None):
        """
        Custom dataset to load images and masks with optional transformations.

        Parameters:
            image_paths (list): List of file paths to images.
            target_paths (list): List of file paths to masks.
            transform (callable, optional): Transformations to apply to the images.
            transform_label (callable, optional): Transformations to apply to the masks.
        """
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.transform_label = transform_label

    def __getitem__(self, index):
        # Load image and mask
        image = imageio.imread(self.image_paths[index])
        image = np.asarray(image, dtype='float32')

        mask = imageio.imread(self.target_paths[index])
        mask = np.asarray(mask, dtype='int64')

        # Ensure consistent random transformations for both image and mask
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        if self.transform:
            image = self.transform(image)

        random.seed(seed)
        torch.manual_seed(seed)

        if self.transform_label:
            mask = self.transform_label(mask)
            # mask = torch.nn.functional.interpolate(
            #     mask,
            #     size=(32, 32),
            #     mode="bilinear",
            #     align_corners=False)
            # mask = F.interpolate(mask, size=(128, 128), mode='bilinear', align_corners=False)
            mask = mask.squeeze(0)  # Squeeze if mask has extra dimension after transformation

        return image, mask

    def __len__(self):
        return len(self.image_paths)


