import matplotlib
matplotlib.use('Agg')
import pylab as plt

import os
import random
import scipy.stats
import numpy as np
import seaborn as sns
from sklearn import linear_model
from spanner import countdown
import grid
import satellite
import zonefinder
import categorical
import importer
import model as mdl

# set plotting defaults
rc = {'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16,
      'axes.titlesize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
      'figure.figsize': (20, 12)}
plt.rcParams.update(rc)
sns.set(rc)

os.system('rm %s/*' % os.path.expanduser('~/Desktop/plots/'))

# load provided flat data file
df = importer.load_data_frame(os.path.join('data', 'hack_day_dataset.txt'))

# split into train/test sets
ids = list(set(df['Field_id']))
random.shuffle(ids)
split = len(ids) / 2
train_ids = ids[:split]
test_ids = ids[split:]
df['Trainset'] = df['Field_id'].map(lambda x: x in train_ids)
df_train = df[df['Trainset']]

df_train = df  # must use full dataset for training / prediction with categorical variables, or else their ordinal conversion will differ

# model leaf area index
model_columns = ['Slope', 'Slope_x_aspect', 'Curvature', 'GDD', 'Pct_clay', 'Pct_silt', 'Pct_sand']
model = mdl.MixedModel(ycol='LAI', xcols=model_columns, xcols_cat=['Region_id', 'Soil_type', 'Crop_guess'],
                       # model=linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials=1000))
                        model=linear_model.LinearRegression())

model.train(df_train)
df['LAI_predicted'] = model.predict(df)

df_test = df[df['Trainset'] == False]

sns.regplot(df['LAI'], df['LAI_predicted'])
plt.savefig(os.path.expanduser('~/Desktop/plots/LAI_regression.png'))



# print results
# correlation heatmap
factors = ['WDRVI', 'LAI', 'LAI_predicted', 'Pct_silt', 'Slope', 'Slope_x_aspect',
           'Aspect', 'Curvature', 'GDD', 'Pct_clay', 'Pct_sand']

corr_heatmap = np.zeros((len(factors), len(factors)))
for i, f1 in enumerate(factors):
    for j, f2 in enumerate(factors):
        if i < j:
            continue
        if i == j:
            corr_heatmap[i, j] = 1
        else:
            r, p_norm = scipy.stats.pearsonr(df[f1], df[f2])
            corr_heatmap[i, j] = corr_heatmap[j, i] = r

plt.figure()
sns.heatmap(corr_heatmap, annot=True, xticklabels=factors, yticklabels=factors, cbar=False)
plt.tight_layout()
plt.savefig(os.path.expanduser('~/Desktop/plots/correlation_heatmap.png'))


# entire region
satellite.retrieve_satellite_image_for_region(df, os.path.join('data', 'satellite', 'region.png'))
satellite.retrieve_satellite_image_for_lat_lon_grid(df, os.path.join('data', 'satellite', 'lat_lon_grid.png'))



contingency_matrix = np.zeros((3, 3))
def print_class_agreement(cm):
    print np.diag(cm) / np.sum(cm, axis=0), sum(np.diag(cm)) / np.sum(cm.flat)


# per-field plots
timer = countdown.timer(len(set(df['Field_id'])), 'Printing pretty plots')
for id, df_field in df.groupby('Field_id'):
    if np.ptp(df_field['LAI']) < 0.5:
        continue


    # if id not in [25, 64, 68, 71, 91, 97, 102, 105, 114, 117, 118, 135, 160, 161, 171, 172, 186]:
    #     continue

    # TODO: why in the world must I do this to prevent warnings when adding new columns??
    df_field = df_field.copy()

    satimg = satellite.retrieve_satellite_image_for_field(df_field, id)
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.imshow(matplotlib.image.imread(satimg), aspect='auto')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    tlow, thigh = np.percentile(df_field['LAI'], [25, 75])

    mz_lai = zonefinder.management_zones(df_field, 'LAI', thresh_low=tlow, thresh_high=thigh, filter_rad=3)
    categorical.heatmap(mz_lai, ncolors=3, transform=False)
    plt.title('Management Zones - LAI')
    plt.subplot(3, 3, 9)
    tlow, thigh = np.percentile(df_field['LAI_predicted'], [25, 75])
    mz_lai_pred = zonefinder.management_zones(df_field, 'LAI_predicted', thresh_low=tlow, thresh_high=thigh, filter_rad=3)
    categorical.heatmap(mz_lai_pred, ncolors=3, transform=False)
    plt.title('Management Zones - LAI predicted')

    contingency_matrix_field = np.zeros((3, 3))
    for zl, zlp in zip(mz_lai.flat, mz_lai_pred.flat):
        if zl != categorical.background:
            contingency_matrix_field[int(zl), int(zlp)] += 1

    contingency_matrix += contingency_matrix_field
    print 'Field %d pct_agreement:' % id
    print_class_agreement(contingency_matrix_field)
    print 'Dataset pct_agreement:'
    print_class_agreement(contingency_matrix)

    soiltypes = list(set(df_field['Soil_base_type'].values))
    df_field['Soil_type_idx'] = df_field['Soil_base_type'].map(lambda x: soiltypes.index(x))

    for factor, cmap, subplot in (('Curvature', 'RdBu_r', 2),
                                    ('Soil_type_idx', 'RdBu_r', 3),
                                    ('Aspect', 'RdBu_r', 4),
                                    ('LAI', 'RdBu_r', 5),
                                    ('Slope', 'Reds', 7),
                                    ('LAI_predicted', 'RdBu_r', 8)):

        plt.subplot(3, 3, subplot)

        if factor == 'Soil_type_idx':
            heatmap, lats, lons = grid.df_to_grid(df_field, factor, default=categorical.background)
            colors = categorical.heatmap(heatmap)

            indexes = sorted(set(df_field[factor]))
            colors_labels = zip(['%s, %s, %s' % (c[0], c[1], c[2]) for c in colors],
                                [soiltypes[idx] for idx in indexes])
            # title = 'Soil types: %s' % (','.join(['\textcolor{%s}{%s}' %
            #                                       (color, label) for color, label in colors_labels]))
            title = (','.join([label for color, label in colors_labels[:2]]))
            if len(title) > 50:
                title = title[:50]
            plt.title(title)
        else:
            if cmap == 'RdBu_r':
                if factor in ('LAI', 'LAI_predicted'):
                    # vmin1, vmax1 = np.percentile(df_field['LAI'], [5, 95])
                    # vmin2, vmax2 = np.percentile(df_field['LAI_predicted'], [5, 95])
                    # vmin = min(vmin1, vmin2)
                    # vmax = max(vmax1, vmax2)
                    # center = (vmax + vmin) / 2
                    vmin, vmax = np.percentile(df_field[factor], [5, 95])
                    center = (vmax + vmin) / 2
                else:
                    vmin, vmax = np.percentile(df_field[factor], [5, 95])
                    center = 0

                empty_cell_value = center
            else:
                vmin = 0
                vmax = None
                center = None
                empty_cell_value = 0

            heatmap, lats, lons = grid.df_to_grid(df_field, factor, default=empty_cell_value)

            sns.heatmap(heatmap, annot=False,
                        vmin=vmin, vmax=vmax,
                        xticklabels=['%1.3f' % l for l in lons],
                        yticklabels=['%1.3f' % l for l in lats],
                        cmap=cmap, center=center, cbar=True)
            plt.title(factor)
            plt.axis('off')


    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Desktop/plots/%1.2f_%s.png' %
                                    (sum(np.diag(contingency_matrix_field)) / np.sum(contingency_matrix_field.flat), id)))
    plt.close()
    timer.tick()

print 'Total dataset pct_agreement: %s' % (sum(np.diag(contingency_matrix)) / np.sum(contingency_matrix.flat))
print_class_agreement(contingency_matrix)
print 'Contingency matrix (rows - LAI, cols - predicted)'
freq_observed = contingency_matrix / np.sum(contingency_matrix.flat)
print freq_observed
print contingency_matrix

marginals = np.matrix([0.25, 0.5, 0.25])
freq_expected = marginals.T.dot(marginals) * np.sum(contingency_matrix.flat)
chsq, pval = scipy.stats.chisquare(contingency_matrix.flat, freq_expected.flat)
print pval
print 'Chi-square goodness of fit: %1.2e' % pval