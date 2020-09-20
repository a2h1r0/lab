import fitbit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

CLIENT_ID = "22BRXS"
CLIENT_SECRET = "a1a90b728db89456cb00a6d1a7fad779"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMkJSWFMiLCJzdWIiOiI4VjQyS0wiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJwcm8gcm51dCByc2xlIiwiZXhwIjoxNjAwNTkzNDEyLCJpYXQiOjE2MDA1NjQ2MTJ9.7F8mqNRbRvXntA6_UZsYOEFjmdZywbU13FCNdVhAx0c"
REFRESH_TOKEN = "035d872d59e8bd026d9c16b2008daa8612be1b1f57387fa25fa125b344134cb8"

# 取得したい日付
DATE = "2020-09-20"

# ID等の設定
authd_client = fitbit.Fitbit(
    CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
# 心拍数を取得（1秒単位）
data_sec = authd_client.intraday_time_series(
    'activities/heart', DATE, detail_level='1sec')  # '1sec', '1min', or '15min'
heart_sec = data_sec["activities-heart-intraday"]["dataset"]
print(heart_sec[:10])
