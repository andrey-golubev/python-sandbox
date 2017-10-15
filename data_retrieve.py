import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_and_prepare(url=None, sep=None, engine=None, normalize=True):
    raw_data = pd.read_csv("armenian_pubs.csv", sep=",", engine="python")
    raw_data.columns = map(str.lower, raw_data.columns) # all columns to lower case
    data = raw_data.drop('timestamp', axis=1)
    data = data.drop('fav_pub', axis=1)
    categorical_cols = [col for col in data.columns if data[col].dtype.name == "object"]
    numerical_cols = [col for col in data.columns if data[col].dtype.name != "object"]
    data = data.fillna(data.median(axis = 0), axis = 0)
    for col in data[categorical_cols]:
        raw_data[col] = data[col].fillna(data[col].describe().top)
        pass
    categorical_descr = data.describe(include=[object])
    binary_cols = [col for col in categorical_cols if categorical_descr[col]['unique'] == 2]
    nonbinary_cols = [col for col in categorical_cols if categorical_descr[col]['unique'] > 2]
    binary_cols = [col for col in categorical_cols if categorical_descr[col]['unique'] == 2]
    nonbinary_cols = [col for col in categorical_cols if categorical_descr[col]['unique'] > 2]
    data.at[data['gender']=='Male', 'gender'] = 0
    data.at[data['gender']=='Female', 'gender'] = 1
    data_nonbinary = pd.get_dummies(data[nonbinary_cols])
    data_nonbinary.columns = map(lambda col: col.split("(")[0], data_nonbinary.columns)
    data_nonbinary.columns = map(lambda col: col.strip(), data_nonbinary.columns)
    data_nonbinary.columns = map(lambda col: col.replace(" ", "_"), data_nonbinary.columns)
    data_nonbinary.columns = map(lambda col: col.replace("'", ""), data_nonbinary.columns)
    data_nonbinary.columns = map(lambda col: col.replace(",", ""), data_nonbinary.columns)
    data_nonbinary.columns.drop_duplicates()
    data_without_y = data[['age', 'income']]
    numerical_cols.remove('wts')
    data_numerical = data_without_y[numerical_cols]
    if normalize is True:
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        data_numerical = pd.DataFrame(min_max_scaler.fit_transform(data_numerical))
    # data_numerical = (data_numerical - data_numerical.mean(axis = 0))/data_numerical.std(axis = 0)
    data_concat = pd.concat((data_numerical, data_nonbinary, data[binary_cols]), axis=1)
    return data_concat, data

def draw_sep_curve_and_reshape(model=None, res = 500, n_dims=2):
    xx0_min, xx0_max = plt.xlim()
    xx1_min, xx1_max = plt.ylim()
    xx0, xx1 = np.meshgrid(np.linspace(xx0_min, xx0_max, res), np.linspace(xx1_min, xx1_max, res))
    yy = model.predict(np.hstack((np.reshape(xx0, (res**n_dims, 1)), np.reshape(xx1, (res**n_dims, 1)))))
    yy = yy.reshape(xx0.shape)
    plt.contourf(xx0, xx1, yy, 1, alpha = 0.25, colors = ('b', 'r'))
    plt.contour(xx0, xx1, yy, 1, colors = 'k')
    plt.xlim((xx0_min, xx0_max))
    plt.ylim((xx1_min, xx1_max))
