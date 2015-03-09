import numpy as np
from sklearn import linear_model
from sklearn import decomposition

def dataframe_to_xy(df, xcols, ycol, xcategorical=[]):
    x = df[xcols].values
    y = df[ycol].values

    for col in xcategorical:
        # transform N categorical values for soil type into N - 1 binary variables so we can use them directly in the regression
        unique_vals = set(df[col].values)
        unique_vals.pop()  # drop one -- doesn't matter which
        for val in unique_vals:
            xx = np.matrix(df[col].map(lambda x: 1 if x == val else 0).values).T
            x = np.hstack((x, xx))

    return x, y


class MixedModel(object):
    def __init__(self, ycol, xcols, xcols_cat, model=linear_model.LinearRegression(), n_princ_comp=None):
        self.ycol = ycol
        self.xcols = xcols
        self.categorical_cols = xcols_cat
        self.model = model
        if n_princ_comp is not None:
            self.pca = decomposition.PCA(n_components=n_princ_comp)
        else:
            self.pca = None



    def train(self, x, y):
        if self.pca is not None:
            self.pca.fit(x)
            x = self.pca.transform(x)
        self.model.fit(x, y)
        print type(self.model), self.model.score(x, y)


    def predict(self, x, y):
        x, y = dataframe_to_xy(df, self.xcols, self.ycol, self.categorical_cols)
        if self.pca is not None:
            x = self.pca.transform(x)

        yhat = self.model.predict(x)
        return yhat


# TODO: would like to build a model that trains a separate estimator for each value of each categorical variable.  Requires careful training set selection
