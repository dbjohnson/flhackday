import os
import importer
import model
from sklearn import ensemble
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
import pylab as plt

# load provided flat data file
df = importer.load_data_frame(os.path.join('data', 'hack_day_dataset_10.txt'))

xcols = ['Slope', 'Slope_x_aspect', 'Curvature', 'Pct_clay', 'Pct_silt', 'Pct_sand']
xcols_cat = ['Region_id', 'Soil_type', 'Crop_guess']
ycol = 'LAI'
X, y = model.dataframe_to_xy(df, xcols, ycol, xcategorical=xcols_cat)

xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size=0.2)
mdl1 = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0, loss='lad')
mdl1.fit(xtrain, ytrain)

def mse(est, X, y):
    yhat = est.predict(X)
    return metrics.mean_squared_error(yhat, y)

print cross_validation.cross_val_score(mdl1, X, y, scoring=mse)


mdl2 = linear_model.LinearRegression()
mdl2.fit(xtrain, ytrain)
print cross_validation.cross_val_score(mdl2, X, y, scoring=mse)



plt.subplot(121)
plt.plot(ytest, mdl1.predict(xtest), 'b.')
plt.subplot(122)
plt.plot(ytest, mdl2.predict(xtest), 'g.')
plt.show()
