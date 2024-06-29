import os
import pickle
import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Load the trained model from a pickle file
def load_model(kind='joblib'):
    """ kind = 'joblib' or 'pkl' """
    
    if kind in ['joblib', 'jbl']:
        with open('./models/model_pipeline.joblib', 'rb') as f:
            model = joblib.load(f)

        
    elif kind in ['pkl', 'pickle']:
        with open('./models/model_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
            
    return model
    
    
def get_metrics(ytest, ypred, model_name=None):
    # Calculate metrics
    f1          = f1_score(ytest, ypred)
    roc_auc     = roc_auc_score(ytest, ypred)
    accuracy    = accuracy_score(ytest, ypred)
    recall      = recall_score(ytest, ypred)
    precision   = precision_score(ytest, ypred)


    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'AUC ROC': roc_auc,
        }
