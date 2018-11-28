from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

import imageio

import torch
from torch.utils import data
from torchvision import transforms
import torchvision.utils as vutils


class RoseDataset(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.images = create_rose_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = self.images[index]
        if self.transform is not None:
            data = self.transform(data)
        return data


def create_rose_ds():
    ds = RoseDataset(transform=transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.CenterCrop(180),
                             transforms.RandomAffine(degrees=45, fillcolor=(255, 255, 255)),
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
    return ds


def create_rose_images():
    return [plot_rose_to_np(k=k, amplitude=10, figsize=2) for k in range(16)]


def rose(delta_steps, k, amplitude):
    theta = np.linspace(0, 2 * np.pi, delta_steps)
    x = amplitude * np.cos(k * theta) * np.cos(theta)
    y = amplitude * np.cos(k * theta) * np.sin(theta)
    return x, y


def plot_rose(k=5, amplitude=10, delta_steps=1000, figsize=20):
    x, y = rose(delta_steps, k, amplitude)
    plt.figure(figsize=(figsize, figsize))
    plt.fill(x, y, 'k', lw=2)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def plot_rose_to_np(k=5, amplitude=10, delta_steps=1000, figsize=5):
    fig = Figure(figsize=(figsize, figsize))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    x, y = rose(delta_steps, k, amplitude)
    ax.fill(x, y, 'k', lw=2)
    ax.axis('equal')
    ax.axis('off')
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8', sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def test():
    plot_rose(k=5, amplitude=10, figsize=5)
    img = plot_rose_to_np(k=5, amplitude=10, figsize=5)
    imageio.imsave('output/rose.png', img)


def test_torch():
    ds = create_rose_ds()
    img = ds[0].permute([1, 2, 0])
    imageio.imsave('output/rose_torch.png', img.cpu().numpy())


def test_torch_loader():
    ds = create_rose_ds()
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)
    batches = cycle(iter(loader))
    vutils.save_image(next(batches).cpu().data, 'output/rose_batch.png', normalize=True, nrow=4)


if __name__ == '__main__':
    test()
    test_torch()
    test_torch_loader()
