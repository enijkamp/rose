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
    def __init__(self, size=1000, transform=None):
        self.transform = transform
        self.images = self.load_images(size, transform)

    @staticmethod
    def load_images(size, transform):
        apply_transform = lambda data: data if transform is None else [transform(d) for d in data]
        sample_data = lambda: apply_transform(create_rose_images())
        images = sample_data()
        while len(images) < size:
            images += sample_data()
        return images[0:size]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def create_rose_ds(size=1000):
    ds = RoseDataset(size, transform=transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.CenterCrop(180),
                             transforms.RandomAffine(degrees=45, fillcolor=(255, 255, 255)),
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
    return ds


def create_rose_ds_single_ch(size=1000):
    ds = RoseDataset(size, transform=transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.Grayscale(num_output_channels=1),
                             transforms.CenterCrop(180),
                             transforms.RandomAffine(degrees=45, fillcolor=(255, )),
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
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


def test(output):
    plot_rose(k=5, amplitude=10, figsize=5)
    img = plot_rose_to_np(k=5, amplitude=10, figsize=5)
    imageio.imsave('{}/rose.png'.format(output), img)


def test_torch(output):
    ds = create_rose_ds()
    img = ds[0].permute([1, 2, 0])
    imageio.imsave('{}/rose_torch.png'.format(output), img.cpu().numpy())


def test_torch_loader(output):
    ds = create_rose_ds()
    loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=0, pin_memory=True, shuffle=True)
    batches = cycle(iter(loader))
    images = next(batches)
    vutils.save_image(images.cpu().data, '{}/rose_batch.png'.format(output), normalize=True, nrow=16)


if __name__ == '__main__':
    output = 'output'
    test(output)
    test_torch(output)
    test_torch_loader(output)
