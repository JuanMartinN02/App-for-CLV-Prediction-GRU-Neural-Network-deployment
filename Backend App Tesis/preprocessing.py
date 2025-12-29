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
import joblib

def preprocess(df):
    data = pd.read_csv(df)
    
    # Making sure Date is in datetime format
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    # Total ammount column
    data['TotalAmount'] = data['Price'] * data['Quantity']

    # Deleting all rows with null values
    data = data.dropna(subset = ['Customer ID'])

    # Eliminating rows with negative quantities
    data = data[data['Quantity'] >= 0]

    data = data[data['Price'] >= 0]

    # Making lists with Customers with heavily outlying behavior
    OutlierCustomersPrice = data[data['Price'] > data['Price'].quantile(0.9995)]['Customer ID'].unique()
    OutlierCustomersQuantity = data[data['Quantity'] > data['Quantity'].quantile(0.9995)]['Customer ID'].unique()

    # Combinning the lists
    outlierCustomers = np.union1d(OutlierCustomersPrice, OutlierCustomersQuantity)

    # Excluding outliers from data
    data = data[~data['Customer ID'].isin(outlierCustomers)]

    # Present value
    presentvalue = data['InvoiceDate'].max()

    # Creating a YearMonth column in the invoice data as we wont use day and hour
    data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')

    # Client by month data
    monthlyData = data.groupby(['Customer ID', 'YearMonth']).agg(
        # Monthly spent
        monthlySpend = ('TotalAmount', 'sum'),

        # Number of invoices
        invoiceQuantity = ('Invoice', 'nunique'),

        # Total item quantity that month
        totalQuantity = ('Quantity', 'sum')
        
    ).reset_index()

    # Definning time period
    startMonth = (presentvalue - pd.DateOffset(months = 22)).to_period('M')
    endMonth = (presentvalue - pd.DateOffset(months = 1)).to_period('M')

    allMonths = pd.period_range(start=startMonth, end=endMonth, freq='M')

    # Indexing by client

    # Unique clients
    customers = monthlyData['Customer ID'].unique()

    # Index client-month
    fullIndex = pd.MultiIndex.from_product([customers, allMonths], names=['Customer ID', 'YearMonth'])

    # Reindex and filling out missing months with 0
    monthlyDataFull = (monthlyData.set_index(['Customer ID', 'YearMonth']).reindex(fullIndex, fill_value=0).reset_index())

    # Making sure monthlyData fits the time window
    monthlyDataFull = monthlyDataFull[(monthlyDataFull['YearMonth'] >= startMonth) & (monthlyDataFull['YearMonth'] <= endMonth)]

    # Clients in monthlyDataFull
    seqCustomers = monthlyDataFull['Customer ID'].unique()

    seqFeatureCols = ['monthlySpend', 'invoiceQuantity', 'totalQuantity']

    # Sorting the time series in a List
    timeOrder = sorted(monthlyDataFull['YearMonth'].unique())

    # Data we'll use to create the X tensor (Input)

    # Unique customers
    customers = monthlyDataFull['Customer ID'].unique()

    # Number of customers
    nClients = len(customers)

    # Number of timesteps (months)
    nTimesteps = len(timeOrder)

    # Number of Features (inputs)
    nFeatures = len(seqFeatureCols)

    # Tensor Init
    X_seq = np.zeros((nClients, nTimesteps, nFeatures), dtype=np.float32)

    # Iteration through the Clients
    for i, customerId in enumerate(customers):

        # Client data
        customerData = monthlyDataFull[
            monthlyDataFull['Customer ID'] == customerId
        ].set_index('YearMonth')

        # Checking cronological order
        customerData = customerData.loc[timeOrder]

        # Feature extraction
        X_seq[i] = customerData[seqFeatureCols].values

    # Log transform
    X_seq = np.log1p(X_seq)

    # Scale with saved scaler
    scaler = joblib.load('robust_scaler_gru.pkl')

    # Function to scale sequence data and not break the 3D shape
    def scaleSequenceData(X, scaler):
        n, t, f = X.shape
        X_flat = X.reshape(n*t, f)
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(n,t,f)

    X_seq = scaleSequenceData(X_seq, scaler)

    return X_seq, customers