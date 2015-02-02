from skimage import measure
from skimage import filter
from skimage import morphology
from scipy import stats
from sklearn import cluster
import numpy as np
import categorical
import grid


def quantize(df, column, percentiles=None, floor=None, ceil=None, num_quant_levels=5):
    if percentiles is not None:
        floor, ceil = np.percentile(df[column], percentiles)

    qtize = np.vectorize(lambda x: int(np.round((num_quant_levels - 1) *
                                                   (min(ceil, max(floor, x)) - floor)/(ceil - floor))))
    return qtize(df[column].tolist())


def connected_components(heatmap_quant, mask, neighbors=8):
    heatmap_quant[mask] = categorical.background
    zones = measure.label(heatmap_quant, neighbors=neighbors, background=categorical.background)
    zones[mask] = categorical.background
    return zones


def kmeans(df, column, clusters=4):
    data = np.matrix(df[column].values).T
    kmeans = cluster.KMeans(init='k-means++', n_clusters=clusters, n_init=10)
    kmeans.fit(data)

    return kmeans.predict(data)


def mode_filter(image, rad=1):
    # hm, can't get the skimage filter to work - roll our own for now
    # selem = morphology.square(width=filter_width)
    # return filter.rank.modal(heatmap_quant, selem=selem)
    rows, cols = image.shape
    filtered = np.zeros((rows, cols))
    for row in xrange(rows):
        for col in xrange(cols):
            if image[row, col] == categorical.background:
                filtered[row, col] = categorical.background
                continue

            values = []
            for r in xrange(max(0, row-rad), min(rows-1, row+rad+1)):
                for c in xrange(max(0, col-rad), min(cols-1, col+rad+1)):
                    if image[r, c] != categorical.background:
                        values.append(image[r, c])

            filtered[row, col] = stats.mstats.mode(values, axis=None)[0]

    filtered = np.vectorize(lambda x: int(x))(filtered)
    return filtered


def management_zones_kmeans(df, column, num_zones, filter_rad=3):
    df['Cluster_indexes'] = kmeans(df, column, clusters=num_zones)
    # remap so cluster indexes are ordered by mean LAI
    cluster_idx_to_mean = dict()
    for cluster_idx, df_cluster in df.groupby('Cluster_indexes'):
        cluster_idx_to_mean[cluster_idx] = df_cluster[column].mean()

    sorted_means = sorted(cluster_idx_to_mean.values())
    cluster_remap = {idx: sorted_means.index(mean) for idx, mean in cluster_idx_to_mean.items()}
    df['Cluster_indexes'] = df['Cluster_indexes'].map(lambda idx: cluster_remap[idx])

    heatmap = grid.df_to_grid(df, 'Cluster_indexes', default=categorical.background, integerize=True)[0]

    if filter_rad > 0:
        filtered = mode_filter(heatmap, rad=filter_rad)
        return filtered
    else:
        return heatmap


def management_zones_quartiles(df, column, filter_rad=3):
    p25, p75 = np.percentile(df[column], [25, 75])
    return management_zones(df, column, thresh_low=p25, thresh_high=p75)



def management_zones(df, column, thresh_low, thresh_high, filter_rad=3):
    df['Cluster_indexes'] = df[column].map(lambda x: 0 if x < thresh_low else (1 if x < thresh_high else 2))
    heatmap = grid.df_to_grid(df, 'Cluster_indexes', default=categorical.background, integerize=True)[0]
    if filter_rad > 0:
        filtered = mode_filter(heatmap, rad=filter_rad)
        return filtered
    else:
        return heatmap

