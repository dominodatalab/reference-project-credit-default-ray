import xgboost as xgb

import pandas as pd
from domino_data_capture.data_capture_client import DataCaptureClient
import uuid

xgc = xgb.Booster(model_file="/mnt/artifacts/tune_best.xgb")

feature_names = ["checking_account_A14", "credit_history_A34", "property_A121", "checking_account_A13", "other_installments_A143", "debtors_guarantors_A103", "savings_A65", "age", "employment_since_A73", "savings_A61"]
predict_names = ['credit']
pred_client = DataCaptureClient(feature_names, predict_names)

def predict_credit(checking_account_A14, credit_history_A34, property_A121, checking_account_A13, other_installments_A143, debtors_guarantors_A103, savings_A65, age, employment_since_A73, savings_A61, customer_id=None):
    
    column_names = ['duration', 'credit_amount', 'installment_rate', 'residence', 'age', 'credits', 'dependents', 'checking_account_A11', 'checking_account_A12', 'checking_account_A13', 'checking_account_A14', 'credit_history_A30', 'credit_history_A31',
                    'credit_history_A32', 'credit_history_A33', 'credit_history_A34', 'purpose_A40', 'purpose_A41', 'purpose_A410', 'purpose_A42', 'purpose_A43', 'purpose_A44', 'purpose_A45', 'purpose_A46', 'purpose_A48', 'purpose_A49', 'savings_A61', 
                    'savings_A62', 'savings_A63', 'savings_A64', 'savings_A65', 'employment_since_A71', 'employment_since_A72', 'employment_since_A73', 'employment_since_A74', 'employment_since_A75', 'status_A91', 'status_A92', 'status_A93', 'status_A94', 
                    'debtors_guarantors_A101', 'debtors_guarantors_A102', 'debtors_guarantors_A103', 'property_A121', 'property_A122', 'property_A123', 'property_A124', 'other_installments_A141', 'other_installments_A142', 'other_installments_A143', 'housing_A151', 
                    'housing_A152', 'housing_A153', 'job_A171', 'job_A172', 'job_A173', 'job_A174', 'telephone_A191', 'telephone_A192', 'foreign_worker_A201', 'foreign_worker_A202']
    
    # Set some default values
    sample_data = [[0.4705882352941176, 0.3685484758446132, 0.3333333333333333, 0.3333333333333333, 0.2857142857142857, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]]
    
    df = pd.DataFrame(sample_data, columns=column_names)
    
    # Override the values with what was passed to the scoring function
    df[["checking_account_A14"]] = checking_account_A14
    df[["credit_history_A34"]] = credit_history_A34
    df[["property_A121"]] = property_A121
    df[["checking_account_A13"]] = checking_account_A13
    df[["other_installments_A143"]] = other_installments_A143
    df[["debtors_guarantors_A103"]] = debtors_guarantors_A103
    df[["savings_A65"]] = savings_A65
    df[["age"]] = age
    df[["employment_since_A73"]] = employment_since_A73
    df[["savings_A61"]] = savings_A61
    
    xgtest = xgb.DMatrix(df)
    predictions = xgc.predict(xgtest)
    pred_class = (predictions > 0.5).astype("int")
    return_dict = {"score" : float(predictions[0]), "class" : int(pred_class)}
    
    if customer_id is None:
        print(f"No Customer ID found! Creating a new one.")
        customer_id = str(uuid.uuid4())
        print(customer_id)
        
    feature_array = list([checking_account_A14, credit_history_A34, property_A121, checking_account_A13, other_installments_A143, debtors_guarantors_A103, savings_A65, age, employment_since_A73, savings_A61])
    predictions = list(map(float,predictions))
    pred_client.capturePrediction(feature_array, list(predictions), event_id=customer_id)
    
    return dict(return_dict)

def main():
    
    """
    
    Test JSON
    
    {
      "data": {
        "checking_account_A14": 0,
        "credit_history_A34": 0,
        "property_A121": 0,
        "checking_account_A13": 0,
        "other_installments_A143": 1,
        "debtors_guarantors_A103": 0,
        "savings_A65": 0,
        "age": 0.285714,
        "employment_since_A73": 1,
        "savings_A61": 1
      }
    }
    
    """
    
    
    pred = predict_credit(0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.285714, 1.000000, 1.000000)
    print(pred)
        
if __name__ == "__main__":
    main()
    
    