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

# Models
from gruModel import ImprovedGRUModel

# Data preprocessing
from preprocessing import preprocess

app = FastAPI()

# Loading pretrained model 
model = ImprovedGRUModel(inputSize=3, hiddenSize=16, numLayers=2, dropout=0.2)
model.load_state_dict(torch.load('gru_model.pt', map_location='cpu'))
model.eval()

@app.post('/predict')
def predict(file: UploadFile):
    # Preprocess the data
    X = preprocess(file.file)
    
    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # Return predictions as a list
    return {'predictions': predictions.tolist()}

