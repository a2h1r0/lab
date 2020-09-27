import fitbit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

CLIENT_ID = "22BRXS"
CLIENT_SECRET = "a1a90b728db89456cb00a6d1a7fad779"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMkJSWFMiLCJzdWIiOiI4VjQyS0wiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNjAxMjE4OTM0LCJpYXQiOjE2MDExOTAxMzR9.i7wnM98_1KGZl1h4-7NVz1Czp5FnMepNW9RYzTGK2aI"
REFRESH_TOKEN = "41140d49392e6d30ee1b4638907d000aa354c1166504f0bef29be95a9552c281"

# 取得したい日付
DATE = "2020-09-20"

# ID等の設定
authd_client = fitbit.Fitbit(
    CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
# 心拍数を取得（1秒単位）
data_sec = authd_client.intraday_time_series(
    'activities/heart', DATE, detail_level='1sec')  # '1sec', '1min', or '15min'
heart_sec = data_sec["activities-heart-intraday"]["dataset"]
print(heart_sec[:200])
