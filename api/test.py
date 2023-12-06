import pandas as pd
import numpy as np
import boto3
from io import StringIO
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import warnings
warnings.filterwarnings("ignore")

# S3 bucket details
bucket_name = 'fooddemandproj'
file_key = 'main.csv'


# Create a Boto3 session with your AWS credentials
session = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
    aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
)

# Create an S3 client using the session
s3 = session.client('s3')

# Get the object from S3
obj = s3.get_object(Bucket=bucket_name, Key=file_key)

# Read the object (which is in bytes) into a pandas DataFrame
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Now you can work with the DataFrame
# print(df.head())


agg_data = df.groupby(['center_id', 'category', 'week']).agg({'num_orders':'sum'})
agg_data

train_ts = agg_data.loc[(10, 'Beverages')]
train_ts

# train model
order = (31, 0, 1)
model = ARIMA(train_ts, order=order).fit()
# model.summary()

# make next week pred
# model.forecast()

print(model.forecast(steps=2))
# joblib.dump(model, 'fooddemand.pkl')