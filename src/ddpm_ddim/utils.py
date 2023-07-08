import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path):
    torchvision.utils.save_image(images, path,
                                 nrow=4,
                                 normalize=True, range=(-1, 1))


def get_data(args):
    dataset = ImageDataset(img_size=args.image_size, dataset_path=args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

class ImageDataset(Dataset):
    def __init__(self, img_size, dataset_path):
        self.train_images = self.listdir(dataset_path)

        transform_list = [
            transforms.Resize(size=[img_size, img_size]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
        ]

        self.transform = transforms.Compose(transform_list)

    def listdir(self, dir_path):
        extensions = ['png', 'jpg', 'jpeg', 'JPG']
        file_path = []
        for ext in extensions:
            file_path += glob(os.path.join(dir_path, '*.' + ext))
        file_path.sort()
        return file_path

    def __getitem__(self, index):
        sample_path = self.train_images[index]
        img = Image.open(sample_path).convert('RGB')
        img = self.transform(img)


        return img

    def __len__(self):
        return len(self.train_images)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)