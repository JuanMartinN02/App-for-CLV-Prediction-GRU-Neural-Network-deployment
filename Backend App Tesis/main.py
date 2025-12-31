# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# for number crunching
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# for dataset management
import pandas as pd

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Extre libraries
from datetime import datetime

# FastAPi
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

# Models
from gruModel import ImprovedGRUModel

# Data preprocessing
from preprocessing import preprocess
import io
import csv

from fastapi.middleware.cors import CORSMiddleware



# Volatile storage for latest prediction, top costumers and segmentation
latestPredictions = []
highValue = []
mediumValue = []
lowValue = []
topCustomer = []

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading pretrained model 
model = ImprovedGRUModel(inputSize=3, hiddenSize=16, numLayers=2, dropout=0.2)
model.load_state_dict(torch.load('gru_model.pt', map_location='cpu'))
model.eval()

@app.post('/predict')
def predict(file: UploadFile):
    # Preprocess the data
    X, customerIDs = preprocess(file.file)
    
    # Convert to torch tensor
    XTensor = torch.tensor(X, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(XTensor).cpu().numpy()

    # Results
    results = [{'CustomerID': int(custID), 'Prediction': float(pred)} for custID, pred in zip(customerIDs, predictions.squeeze())]

    # Save in memory
    latestPredictions.clear()
    latestPredictions.extend(results)
    
    # Return predictions as a list
    return {'predictions': results}

# Function that returns top N customers in the latest prediction
@app.get('/topCustomers')
def getTopCustomers(topN: int=50):

    if not latestPredictions:
        return {'error': 'No predictions available. Run /predict first.'}
    else:
        topCustomersTemp = sorted(latestPredictions, key=lambda x: x['Prediction'], reverse=True)[:topN]

        # Save in memory
        topCustomer.clear()
        topCustomer.extend(topCustomersTemp)

        return topCustomersTemp
    

# Funtion that return the metrics of the latest prediction
@app.get('/metrics')
def getMetrics():
    if not latestPredictions:
        return {'error': 'No predictions available. Run /predict first.'}
    else:
        values = np.array([pred['Prediction'] for pred in latestPredictions])

        return {
            'count': len(values),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'p90': float(np.percentile(values, 90)),
            'max': float(values.max()),
        }


# Function for customer segmentation
@app.get('/segment')
def getSegmentation():
    if not latestPredictions:
        return {'error': 'No predictions available. Run /predict first.'}
    else:
        highValueTemp = []
        mediumValueTemp = []
        lowValueTemp = []

        values = np.array([pred['Prediction'] for pred in latestPredictions])

        # Definning quantile values
        p75 = np.quantile(values, 0.75)
        p90 = np.quantile(values, 0.90)

        # Segmentation
        for customer in latestPredictions:
            if customer['Prediction'] >= p90:
                highValueTemp.append(customer)
            elif customer['Prediction'] >= p75:
                mediumValueTemp.append(customer)
            else:
                lowValueTemp.append(customer)

        # Save in memory
        highValue.clear()
        mediumValue.clear()
        lowValue.clear()
        highValue.extend(highValueTemp)
        mediumValue.extend(mediumValueTemp)
        lowValue.extend(lowValueTemp)

        return {
            'segmentation': {
                'high_value': {
                    'count': len(highValue),
                    'threshold': f'>= {p90:.4f} £',
                },
                'medium_value': {
                    'count': len(mediumValue),
                    'threshold': f'{p75:.4f} £ - {p90:.4f} £',
                },
                'low_value': {
                    'count': len(lowValue),
                    'threshold': f'< {p75:.4f} £',
                }
            },
            'thresholds': {
                'p75': float(p75),
                'p90': float(p90)
            }
        }


# Function to export the lastest prediction as a CSV
@app.get('/export')
def exportToCSV():
    if not latestPredictions:
        return {'error': 'No predictions available. Run /predict first.'}
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['CustomerID', 'Prediction'])
    writer.writeheader()
    writer.writerows(latestPredictions)
    
    # Return as downloadable file
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )

# Function to export the High value customers list as a CSV
@app.get('/exportHigh')
def exportToCSV():
    if not highValue:
        return {'error': 'No predictions available. Run /predict first.'}
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['CustomerID', 'Prediction'])
    writer.writeheader()
    writer.writerows(highValue)
    
    # Return as downloadable file
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=HighValueCustomers.csv"}
    )

# Function to export the Medium value customers list as a CSV
@app.get('/exportMedium')
def exportToCSV():
    if not mediumValue:
        return {'error': 'No predictions available. Run /predict first.'}
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['CustomerID', 'Prediction'])
    writer.writeheader()
    writer.writerows(mediumValue)
    
    # Return as downloadable file
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=MediumValueCustomers.csv"}
    )

# Function to export the Low value customers list as a CSV
@app.get('/exportLow')
def exportToCSV():
    if not lowValue:
        return {'error': 'No predictions available. Run /predict first.'}
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['CustomerID', 'Prediction'])
    writer.writeheader()
    writer.writerows(lowValue)
    
    # Return as downloadable file
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=LowValueCustomers.csv"}
    )

