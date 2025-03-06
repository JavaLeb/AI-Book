import os
import pandas as pd


def load_data(data_dir):
    """Load the GEFCom 2014 energy load data"""

    energy = pd.read_csv(
        os.path.join(data_dir, "energy.csv"), parse_dates=["timestamp"]
    )

    # Reindex the dataframe such that the dataframe has a record for every time point
    # between the minimum and maximum timestamp in the time series. This helps to
    # identify missing time periods in the data (there are none in this dataset).

    energy.index = energy["timestamp"]
    energy = energy.reindex(
        pd.date_range(min(energy["timestamp"]), max(energy["timestamp"]), freq="h")
    )
    energy = energy.drop("timestamp", axis=1)

    return energy


import numpy as np

data_dir = r'E:\pycharm_workspace\AI-Book\时间序列\data'

ts_data_load = load_data(data_dir)[["load"]]

from pandas.plotting import lag_plot
import matplotlib.pyplot as plt

# plt.figure()

# lag_plot(ts_data_load)


plt.figure()
ts_data_load_subset = ts_data_load["2014-08-01 00:00:00":"2014-08-01 04:00:00"]
print(ts_data_load_subset)

from pandas.plotting import autocorrelation_plot

ax = autocorrelation_plot(ts_data_load_subset)

# 自相关
data = ts_data_load_subset['load']
autocorr_1 = data.autocorr(lag=1)
print(autocorr_1)

n = 5
h = 1
data = ts_data_load_subset
x_mean = np.mean(data)  # 2601.8
c0 = np.sum((data - x_mean) ** 2) / n  # 6294.96
b = (np.asarray(data[: n - h] - x_mean) * np.asarray(data[h:] - x_mean)).sum() / n
r=b/ c0
print('rr=', r)

# [2566.   2606.75]
xi = data[:-1]
xi_bar = xi - xi.mean()
yi = data[1:]
yi_bar = yi - yi.mean()
# [[ 1947.33333333  2913.33333333], [ 2913.33333333 10328.25      ]]
# [[ 55.   -31.   -40.    16.  ], [138.25  14.25 -71.75 -80.75]]
numerator = np.sum(np.asarray(xi_bar) * np.asarray(yi_bar))
da = np.sum(xi_bar ** 2)
db = np.sum(yi_bar ** 2)
denominator = np.sqrt(da) * np.sqrt(db)
r = numerator / denominator
print('r=', r)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(ts_data_load_subset)
