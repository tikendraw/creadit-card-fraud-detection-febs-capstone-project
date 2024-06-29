import pandas as pd

from utils import load_model, get_metrics
from sklearn.pipeline import Pipeline
import sys

model = load_model('joblib')


def give_x_y(df, frac=1.0):
    
    df= df.sample(frac=frac)
    df = df.drop('Time', axis=1)
    x = df.drop("Class",axis=1)
    y = df['Class']
    return x,y

        
    
if __name__=='__main__':

    csv_file = sys.argv[1]
    frac = sys.argv[2] if len(sys.argv)>2 else 1.0
    df = pd.read_csv(csv_file)
    x,y = give_x_y(df, frac)
    ypred= model.predict(x)
    
    print(get_metrics(ytest=y.to_numpy(),ypred=ypred))
