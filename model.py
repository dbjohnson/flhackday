import numpy as np
from sklearn import linear_model

def _dataframe_to_xy(df, xcols, ycol, xcategorical=[]):
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
    def __init__(self, ycol, xcols, xcols_cat, model=linear_model.LinearRegression()):
        self.ycol = ycol
        self.xcols = xcols
        self.categorical_cols = xcols_cat
        self.model = model


    def train(self, df):
        x, y = _dataframe_to_xy(df, self.xcols, self.ycol, self.categorical_cols)
        self.model.fit(x, y)
        print type(self.model), self.model.score(x, y)


    def predict(self, df):
        x, y = _dataframe_to_xy(df, self.xcols, self.ycol, self.categorical_cols)
        yhat = self.model.predict(x)
        return yhat


# TODO: would like to build a model that trains a separate estimator for each value of each categorical variable.  Requires careful training set selection
