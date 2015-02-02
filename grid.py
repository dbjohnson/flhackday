import numpy as np
import categorical


def __dedupe_preserve_order(_list):
    seen = set()
    seen_add = seen.add
    return [x for x in _list if not (x in seen or seen_add(x))]


def df_to_grid(df, factor, default=0, integerize=False):
    return lat_lon_z_to_grid(df['Latitude'], df['Longitude'], df[factor],
                             default=default, integerize=integerize)


def lat_lon_z_to_grid(lats, lons, zs, default=0, integerize=False):
    deduped_lats = __dedupe_preserve_order(lats)
    deduped_lons = __dedupe_preserve_order(lons)
    heatmap = np.ones((len(deduped_lats), len(deduped_lons))) * default
    for (lat, lon, z) in zip(lats, lons, zs):
        heatmap[deduped_lats.index(lat), deduped_lons.index(lon)] = z

    if integerize:
        heatmap = np.vectorize(lambda x: int(x))(heatmap)

    return heatmap, deduped_lats, deduped_lons
