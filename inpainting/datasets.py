import numpy
import torch
import torchvision
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from patchify import patchify
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

from dataclasses import dataclass


def patchslice(img, shape):
    return patchify(numpy.array(img), (shape, shape, img.shape[-1]), shape).reshape(
        -1, shape * shape
    )


@dataclass
class PatchedImageDataset(Dataset):
    data: numpy.typing.NDArray[float]
    clusters: int
    shape: int
    dropout: float = 0.15
    unimask: bool = True
    embeddings: bool = False

    def __post_init__(self):
        patches = numpy.vstack([patchslice(x, shape=self.shape) for x in self.data])
        self.kclassifier = KMeans(n_clusters=self.clusters).fit(patches)
        self.itop = lambda i: self.kclassifier.cluster_centers_[i]
        self.ptoi = lambda p: self.kclassifier.predict(p)
        self.mask_id = self.clusters
        self.tokens = self.clusters + 1  # normal tokens + mask token

        _, counts = numpy.unique(self.ptoi(patches), return_counts=True)
        self.distribution = counts / len(patches)

        centroids = self.itop(numpy.arange(self.clusters))
        centroids = numpy.vstack([centroids, numpy.zeros_like(centroids[0])])
        self.euclideans = cdist(centroids, centroids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        p = self.ptoi(patchslice(x, shape=self.shape))
        mask = self.mask_id
        p = torch.from_numpy(p).long()
        q = p.clone()

        if self.embeddings:
            q = torch.from_numpy(self.euclideans[p]).float()
            mask = torch.zeros_like(q[0])

        if self.unimask:
            num_to_mask = max(1, round(self.dropout * len(p)))  # Ensure at least 1 token is masked
            mask_positions = torch.randperm(len(p))[:num_to_mask] 
        else:
            mask_positions = torch.rand(p.shape) < self.dropout
        
        q[mask_positions] = mask
        return dict(input_ids=q, labels=p)

    def plot_sample(self, patches, shape=None, scale=1):
        shape = shape or self.data.shape[1] // self.shape
        patches = patches.reshape(shape, shape)
        fig, axes = plt.subplots(
            shape, shape, figsize=(6 * scale, 6 * scale), facecolor="black"
        )
        for arow, prow in zip(axes, patches):
            for ax, patch in zip(arow, prow):
                ax.axis("off")
                try:
                    ax.imshow(
                        self.itop(patch).reshape(self.shape, self.shape),
                        vmin=0,
                        vmax=255,
                        cmap="Blues",
                    )
                except IndexError:
                    ax.imshow(
                        numpy.zeros((self.shape, self.shape)),
                        vmin=0,
                        vmax=255,
                        cmap="binary_r",
                    )


class MNIST(PatchedImageDataset):
    def __init__(self, clusters=400, frac=1.0, train=True, embeddings=False, unimask=False, shape=2):
        data = torchvision.datasets.MNIST("./data", train=train, download=True).data
        size = int(min(len(data) * frac, len(data)))
        self.shape = shape 
        super().__init__(
            data=data[:size][:, :, :, None],
            clusters=clusters,
            shape=shape,
            embeddings=embeddings,
            unimask=unimask,
        )
