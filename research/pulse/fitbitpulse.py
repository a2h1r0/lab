import fitbit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

CLIENT_ID = "22BRXZ"
CLIENT_SECRET = "2f44edc1872d6976b194251d466a4d69"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMkJSWFoiLCJzdWIiOiI4VjU4TUQiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNjAwNDY3NTYzLCJpYXQiOjE2MDA0Mzg3NjN9.X1TmLy2iVXh7L58UXWZxG4iqfkg5SbM5QTubrqOWezM"
REFRESH_TOKEN = "12f61c444bc95c7b04381eefab1853453e2c1640bbd4b1cfd3a7b94bebc4ca25"

# 取得したい日付
DATE = "2020-09-17"

# ID等の設定
authd_client = fitbit.Fitbit(
    CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
# 心拍数を取得（1秒単位）
data_sec = authd_client.intraday_time_series(
    'activities/heart', DATE, detail_level='1sec')  # '1sec', '1min', or '15min'
heart_sec = data_sec["activities-heart-intraday"]["dataset"]
print(heart_sec[:10])
