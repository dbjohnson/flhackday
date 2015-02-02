from matplotlib import colors as clr
import pylab as plt
import seaborn as sns
import numpy as np

background = -1<<30


def heatmap(image, ncolors=None, transform=True):
    distinct_values = list(sorted(np.unique(image)))
    if background in distinct_values:
        distinct_values.remove(background)

    if ncolors is None:
        ncolors = len(distinct_values)

    colors = sns.color_palette("hls", ncolors)
    background_color = [0.95, 0.95, 0.95]
    cmap_colors = colors[:]
    cmap_colors.insert(0, background_color)
    cmap = clr.ListedColormap(cmap_colors)

    display_im = []
    if transform:
        for row in image:
            display_im.append([background_color if val == background
                               else colors[distinct_values.index(val) % len(colors)] for val in row])
    else:
        for row in image:
            display_im.append([background_color if val == background
                               else colors[val % len(colors)] for val in row])


    rows, cols = image.shape
    plt.imshow(display_im, interpolation='none', aspect='auto', extent=[0, cols, 0, rows], cmap=cmap)
    plt.colorbar(ticks=[])
    plt.grid(ls='solid', color='w')
    plt.xticks(range(cols), [])
    plt.yticks(range(rows), [])
    return colors


def get_unique_values_and_index(values):
    distinct_values = list(sorted(np.unique(values)))
    return distinct_values, [distinct_values.index(v) for v in values]