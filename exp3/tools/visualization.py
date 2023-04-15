import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_features(feature_maps):
    feature_maps = feature_maps.cpu().numpy()
    fig, axes = plt.subplots(nrows=16, ncols=32, figsize = (16, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.02)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        img = feature_maps[i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 1e-8)
        im = ax.imshow(img, cmap='gray')

    fig.colorbar(im, ax = axes.ravel().tolist())
    plt.show()