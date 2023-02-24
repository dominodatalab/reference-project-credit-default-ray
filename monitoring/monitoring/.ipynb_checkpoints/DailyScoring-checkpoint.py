import pandas as pd
import numpy as np
import random
import math
import pickle
import json
import os
import requests
import datetime
import boto3
from botocore.exceptions import NoCredentialsError
from domino.training_sets import TrainingSetClient, model
import urllib.request
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from random import *

#set DMM vars
bucket = 'credit-risk-monitor'

#Load in data
# Set for reprudicibility
np.random.seed(1234)

# Setup paths and filenames
DATA_PATH = "/domino/datasets/local/" + os.environ.get("DOMINO_PROJECT_NAME") # Location of the Credit Card Dataset
TEST_SET = DATA_PATH + "/data/test_data.csv"
SCORING_SET = DATA_PATH + "/data/scoring_data_0.csv"

test_df = pd.read_csv(TEST_SET, sep=",")

X = test_df.loc[:, test_df.columns != "credit"]
y = test_df["credit"]

def generate(n_chunks, chunk_size, file_name_prefix):

    for chunk in range(n_chunks):

        print("Generating chunk {}".format(chunk))
        df_chunk = pd.DataFrame().reindex(columns=X.columns)

        while (df_chunk.shape[0] < chunk_size):
            X_sample = X.sample(100, random_state=chunk)
            y_sample = y[X_sample.index]

            sampler = SMOTEN(random_state=chunk)
            X_balanced, y_balanced = sampler.fit_resample(X_sample, y_sample)

            df_chunk = pd.concat([df_chunk, pd.concat([X_balanced, y_balanced], axis=1)], axis=0)


        df_chunk["credit"] =df_chunk["credit"].astype(int)
        df_chunk.head(chunk_size).to_csv(file_name_prefix + str(chunk) + ".csv", sep=",", header=True, index=False)
        
    print("Number of samples in the generated set is: {:,}".format(n_chunks * chunk_size))

# Generate training data
generate(1, randint(100, 300), DATA_PATH + "/data/scoring_data_")

#read in the new scoring data
df_inf = pd.read_csv(SCORING_SET, sep=",")

#set up clean customer_ids
setup_ids = list(range(0, df_inf.shape[0]))
ids = list()
for i in setup_ids:
    ids.append(str(datetime.date.today())+'_'+str(setup_ids[i]))
    
df_inf['customer_id']=ids    

# Loop through the records and send them to the API
print('Sending {} records to model API endpoint for scoring'.format(df_inf.shape[0]))
#Set up dictionaries and lists for loops
setup_dict = {}
scoring_request = {}
results = list()

# Use the subset of features for the API call
feature_names = ["checking_account_A14", "credit_history_A34", "property_A121", "checking_account_A13", "other_installments_A143", "debtors_guarantors_A103", "savings_A65", "age", "employment_since_A73", "savings_A61", "customer_id"]
inputs = df_inf[feature_names]

for n in range(inputs.shape[0]):
    for i in list(inputs.columns):
        setup_dict.update({i :list(inputs[n:n+1].to_dict().get(i).values())[0]})
        scoring_request = {'data' : setup_dict}
        
        
        response = requests.post("https://demo2.dominodatalab.com:443/models/63f889a99fb0fd477f3a599e/latest/model",
    auth=(
        "B0HjcRkGR9YqicRzxRIN08rc2hor1vsZdPoR5mFF1BvvbR1iFRZZKRBgb8RWvGNv",
        "B0HjcRkGR9YqicRzxRIN08rc2hor1vsZdPoR5mFF1BvvbR1iFRZZKRBgb8RWvGNv"
    ),
        json=scoring_request
    )
    results.append(response.json().get('result'))

print('Scoring complete')

# Prepare the ground truth data. This will simply be the customer_id to match with and the credit classification
df_ground_truth=df_inf[['customer_id', 'credit']].rename({'customer_id': 'event_id', 'credit' : 'credit_GT'}, axis=1)
print(df_ground_truth.shape[0]==inputs.shape[0])
print((df_ground_truth.event_id==inputs.customer_id).sum()==df_ground_truth.shape[0])

# Create the /data folder in the dataset if it hasn't already been created.
if not os.path.isdir(DATA_PATH + "/data/gt_data"):
    os.makedirs(DATA_PATH + "/data/gt_data")
    
# Save the ground truth data to the dataset
gt_file_name = str('GT_Data_') + os.environ['DOMINO_PROJECT_ID'] + str(datetime.date.today())+str('.csv')
gt_file_path = str(DATA_PATH + "/data/gt_data/")+gt_file_name
df_ground_truth.to_csv(gt_file_path, index=False)

# upload the file to S3 for Monitor to consume it later. There is a separate script for registering the ground truth data from S3 to monitor - GT_registration.py
def s3_upload(local_file, bucket):
    s3 = boto3.client('s3', 
                      aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                      aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    
    s3_file_name = '{}'.format(os.path.basename(local_file))
    
    try:
        s3.upload_file(local_file, bucket, s3_file_name)
        print(str(s3_file_name) + " Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    
s3_upload(gt_file_path, bucket)

print('Data Uploaded to s3 bucket at s3://{}/{}'.format(bucket, gt_file_name))
print('Done!')