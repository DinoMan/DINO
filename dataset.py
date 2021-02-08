from torch.utils.data import Dataset
from torchvision import transforms
import utils
from PIL import Image
import os
import random


class MatchedImageDataset(Dataset):
    def __init__(self, folders, img_size, mirror=False, random_crop=False, ext=None):
        self.folders = folders
        self.files = utils.list_matching_files(folders, ext=[None, ext])
        self.mirror = mirror
        self.cropping = utils.RandomCrop()
        self.random_crop = random_crop
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.aug_image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, sample_idx):
        image_list = []

        for i, f in enumerate(self.folders):
            image_list.append(Image.open(os.path.join(f, self.files["dirs"][sample_idx], self.files["files"][sample_idx] + self.files["exts"][sample_idx][i])).convert("RGB"))

        if self.random_crop and random.randint(0, 1) == 0:
            image_list = self.cropping(image_list)  # Perform augmentation through random cropping

        if self.mirror:
            if random.randint(0, 1) == 0:  # Perform horizontal flipping half of the times
                image_list = [self.image_transform(img) for img in image_list]
            else:
                image_list = [self.aug_image_transform(img) for img in image_list]
        else:
            image_list = [self.image_transform(img) for img in image_list]

        return image_list

    def __len__(self):
        return len(self.files["files"])
