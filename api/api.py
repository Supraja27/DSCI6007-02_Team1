from fastapi import FastAPI
import pandas as pd
import numpy as np
from glob import glob
import boto3
from io import StringIO
from pydantic import BaseModel
import joblib
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import warnings
warnings.filterwarnings("ignore")

bucket_name = 'fooddemandproj'
file_key = 'main.csv'

class PredictionRequest(BaseModel):
    center_id: int
    category: str
    num_weeks: int



# Create a Boto3 session with your AWS credentials
session = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
    aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
)

# Get data from s3
s3 = session.client('s3')
obj = s3.get_object(Bucket=bucket_name, Key=file_key)
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Define Helper Functions
def dump(model, request):
    timestamp = pd.Timestamp.now().isoformat()
    model_directory = 'models'
    model_name = f'center{request.center_id}_{request.category}'
    filepath = os.path.join(model_directory, f'{timestamp}_{model_name}.pkl')
    model.save(filepath)

def load(request):
    model_directory = 'models'
    model_name = f'center{request.center_id}_{request.category}'
    pattern = os.path.join(model_directory, f'*{model_name}.pkl')
    model_path = sorted(glob(pattern))[-1]
    return joblib.load(model_path)

def model_exists(request):
    model_directory = 'models'
    model_name = f'center{request.center_id}_{request.category}'
    pattern = os.path.join(model_directory, f'*{model_name}.pkl')
    if glob(pattern):
        return True
    return False

app = FastAPI()

@app.post("/predict")
def make_prediction(request: PredictionRequest):
    # Aggregate data and extract relevant time series
    agg_data = df.groupby(['center_id', 'category', 'week']).agg({'num_orders':'sum'})
    if (request.center_id, request.category) in agg_data.index:
        # agg_data.loc[(10, 'Beverages')]
        train_ts = agg_data.loc[(request.center_id, request.category)]
    else:
        return {"error": "Data not found for the given center_id and category"}
    
    # Determine the last week of the training data
    last_week = train_ts.index[-1]

    # Use existing model for forecast
    if model_exists(request):
        model = load(request)

        forecast = model.forecast(steps=request.num_weeks)

        # Generate the weeks for the forecast
        last_week = int(train_ts.index[-1])
        forecast_weeks = [last_week + i for i in range(1, request.num_weeks + 1)]

        # Convert numpy.float64 to float for JSON serialization
        forecast = [float(value) for value in forecast]

        # Combine the forecast with the weeks
        forecast_result = [{"week": week, "prediction": prediction} for week, prediction in zip(forecast_weeks, forecast)]
    else:
        # Fit the ARIMA model using the best hyperparameters
        # specify ar & ma parameters
        ar_params = list(range(1, 32, 10))
        ma_params = list(range(0, 4, 1))

        best_mae = 9999 # select high number
        best_model_hyperparam = tuple()

        # loop through diff combinations of ar & ma params and fit model
        for ar_param in ar_params:
            # Inner loop: Iterate through possible values for `ma_param`
            for ma_param in ma_params:
                # Combination of hyperparameters for model
                order = (ar_param, 0, ma_param)
                model = ARIMA(train_ts, order=order).fit()
                # calculate mae
                model_pred = model.predict()
                current_mae = mean_absolute_error(train_ts, model_pred)

                # store best params
                if current_mae <= best_mae:
                    best_mae = current_mae
                    best_model_hyperparam = order

        # Use best model
        model = ARIMA(train_ts, order=best_model_hyperparam).fit()

        forecast = model.forecast(steps=request.num_weeks)

        # Generate the weeks for the forecast
        last_week = int(train_ts.index[-1])
        forecast_weeks = [last_week + i for i in range(1, request.num_weeks + 1)]

        # Convert numpy.float64 to float for JSON serialization
        forecast = [float(value) for value in forecast]

        # Combine the forecast with the weeks
        forecast_result = [{"week": week, "prediction": prediction} for week, prediction in zip(forecast_weeks, forecast)]

        # Dump Model
        dump(model, request)

    return {"forecast": forecast_result}