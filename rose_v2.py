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
                             transforms.CenterCrop(300),
                             transforms.RandomAffine(degrees=45, scale=[0.5, 1.0]),
                             transforms.CenterCrop(70),
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
    return ds


def create_rose_images():
    k_s = [k for k in range(16)]
    fg_c_s = ['b', 'r', 'c', 'm', 'y', 'w']
    bg_c_s = [(0, .5, 0), (0, .5, .3), (0, 102/255, 0), (9/255, 51/255, 0)]
    images = []
    for k in k_s:
        for fg_c in fg_c_s:
            for bg_c in bg_c_s:
                images.append(plot_rose_to_np(k=k, amplitude=10, figsize=2, color_fg=fg_c, color_bg=bg_c))

    return images


def rose(delta_steps, k, amplitude):
    theta = np.linspace(0, 2 * np.pi, delta_steps)
    x = amplitude * np.cos(k * theta) * np.cos(theta)
    y = amplitude * np.cos(k * theta) * np.sin(theta)
    return x, y


def plot_rose(k=5, amplitude=10, delta_steps=1000, figsize=20, color_fg='k', color_bg='w'):
    x, y = rose(delta_steps, k, amplitude)
    fig = plt.figure(figsize=(figsize, figsize), facecolor=color_bg)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axis('equal')
    plt.xlim([-amplitude, amplitude])
    plt.ylim([-amplitude, amplitude])
    plt.fill(x, y, color_fg, lw=2)
    plt.show()


def plot_rose_to_np(k=5, amplitude=10, amplitude_bg_factor=3, delta_steps=1000, figsize=5, color_fg='w', color_bg='k'):
    fig = Figure(figsize=(figsize, figsize), facecolor=color_bg)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.axis('equal')
    ax.set_xlim([-amplitude_bg_factor*amplitude, amplitude_bg_factor*amplitude])
    ax.set_ylim([-amplitude_bg_factor*amplitude, amplitude_bg_factor*amplitude])
    x, y = rose(delta_steps, k, amplitude)
    ax.fill(x, y, color_fg, lw=2)
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
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), num_workers=0, pin_memory=True, shuffle=True)
    batches = cycle(iter(loader))
    vutils.save_image(next(batches).cpu().data, '{}/rose_batch.png'.format(output), normalize=True, nrow=8)


if __name__ == '__main__':
    output = 'output_v2'
    test(output)
    test_torch(output)
    test_torch_loader(output)

