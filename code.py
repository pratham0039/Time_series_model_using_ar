import inspect
import time
import warnings
from pprint import PrettyPrinter

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")


client = MongoClient(host="localhost",port = 27017)
db = client["air-quality"]
dar = db["dar-es-salaam"]

sites = dar.distinct("metadata.site")
sites

result = dar.find_one({})
pp.pprint(result)


result = dar.aggregate(
    [
        {"$group": {"_id": "$metadata.site", "count": {"$count":{}}}}
    ]

)
readings_per_site = list(result)
readings_per_site


def wrangle(collection):
    results = collection.find(
        {"metadata.site": 11, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read results into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Dar_es_Salaam")

    # Remove outliers
    df = df[df["P2"] < 100]

    # Resample and forward-fill
    y = df["P2"].resample("1H").mean().fillna(method="ffill")
    
    return y

y = wrangle(dar)
y.head()
fig, ax = plt.subplots(figsize=(15, 6))
y.plot(ax=ax)
fig, ax = plt.subplots(figsize=(15, 6))
y.rolling(168).mean().plot(ax=ax),

fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y,ax=ax);

fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y,ax=ax)
cutoff_test = int(len(y)*0.9)
y_train = y.iloc[:cutoff_test]
y_test = y.iloc[cutoff_test:]
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean]*len(y_train)
mae_baseline = mean_absolute_error(y_train,y_pred_baseline)

print("Mean P2 Reading:", y_train_mean)
print("Baseline MAE:", mae_baseline)



p_params = range(1, 31)
maes = []
for p in p_params:
    model=AutoReg(y_train, lags=p).fit()
    res = model.predict().dropna()
    mae = mean_absolute_error(y_train[p:],res)
    maes.append(mae)
    
    pass
mae_series = pd.Series(maes, name="mae", index=p_params)
mae_series

best_p =  AutoReg(y_train,lags=(1,30)).fit()
best_model = best_p

y_pred = model.predict().dropna()
y_train_resid = y_train-y_pred
y_train_resid.name = "residuals"
y_train_resid.head()

fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid,ax=ax)


#predict using history

y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model=AutoReg(history,lags=28).fit()
    print(i)
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history =history.append(y_test[next_pred.index])
    
    
y_pred_wfv.name = "prediction"
y_pred_wfv.index.name = "timestamp"
y_pred_wfv.head()

df_pred_test = pd.DataFrame({"y_test":y_test,"y_pred_wfv":y_pred_wfv})
fig = px.line(df_pred_test)
fig.update_layout(
    title="Dar es Salaam, WFV Predictions",
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
)


fig.show()




