# rose
Procedural dataset generator of flowers with k pedals using the rose curve (Guido Grandi).

![rose](output/rose_batch.png)

PyTorch:

```python
ds = RoseDataset(transform=transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.CenterCrop(180),
                         transforms.RandomAffine(degrees=45, fillcolor=(255, 255, 255)),
                         transforms.Resize(32),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ]))
```
