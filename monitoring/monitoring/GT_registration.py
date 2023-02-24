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
 
#set DMM vars - these come from the Monitor UI
data_source = 'Credit-Risk-Monitor'
model_id='63f87cd775b59272c171cc29'
 
dmm_api_key = os.environ['DMM_API_KEY']
 
 
gt_file_name = str('GT_Data_') + os.environ['DOMINO_PROJECT_ID'] + str(datetime.date.today())+str('.csv')
 
print('Registering {} From S3 Bucket in DMM'.format(gt_file_name))
 
#create GT payload    
 
#Set up call headers
headers = {
           'X-DMM-API-KEY': dmm_api_key,
           'Content-Type': 'application/json'
          }
 
 
 
ground_truth_payload = """
{{
"datasetDetails": {{
        "name": "{0}",
        "datasetType": "file",
        "datasetConfig": {{
            "path": "{0}",
            "fileFormat": "csv"
        }},
        "datasourceName": "{1}",
        "datasourceType": "s3"
    }}
}}
""".format(gt_file_name, data_source)
 
#Define api endpoint
ground_truth_url = "https://demo2.dominodatalab.com/model-monitor/v2/api/model/{}/register-dataset/ground_truth".format(model_id)
 
#Make api call
ground_truth_response = requests.request("PUT", ground_truth_url, headers=headers, data = ground_truth_payload)
 
#Print response
print(ground_truth_response.text.encode('utf8'))
 
print('DONE!')