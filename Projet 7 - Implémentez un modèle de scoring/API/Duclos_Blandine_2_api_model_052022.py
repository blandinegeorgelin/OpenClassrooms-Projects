# Library import
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from os.path import join
import pickle
import dill
import gzip
import bz2

app = FastAPI(title = "Prediction Model API for Loan"
              , description = "A simple API that use machine learning model to predict the probability of difficulties paiements loan."
              , version = "0.1")

class Loan(BaseModel):
    LoanID: int
    
@app.get("/")
def root():
    return {"message": "Welcome on the Loan Classification App"}
    
@app.get("/features_importances")
def features_importances_globals():
    # load training model
    try: 
        model = pickle.load(bz2.open(join("dataviz","classifier.pkl"), "rb")) #abspath
    except:
        raise 'You must train the model first.'
    
    # features importances : already charge in the opening app
    try:
        feat_imp = model.steps[-1][1].feature_importances_
    except:
        feat_imp = model.steps[-1][1].coef_[0]
    feat_names = model.feature_names_in_
    del model

    df_feat_imp = pd.DataFrame(feat_imp
                                , columns = ["Coefficients"]
                                , index = feat_names
                                )
    del feat_imp, feat_names

    df_feat_imp = df_feat_imp.sort_values('Coefficients', ascending = False).head(20)

    return {"feat_imp": df_feat_imp}

@app.post('/predict/{id_loan}')
async def predict_solvent(loan: Loan):
    # chargement des données explicatives
    try: 
        with bz2.open(join("dataviz",'data_test.csv')) as f:
            row_count = sum(1 for row in f) - 1
    except:
        raise 'No application train data is available, please check the csv file.'
    listSizeRow = np.arange(0,row_count,10000)
    del row_count
    idx = -1
    for i in np.arange(0,len(listSizeRow)):
        df = pd.read_csv(join("dataviz",'data_test.csv'), compression = "bz2", skiprows = listSizeRow[i], nrows = 10000)
        if loan.LoanID in df['SK_ID_CURR'].unique().tolist():
            dataId = df[df['SK_ID_CURR'] == loan.LoanID]
            dataId = dataId.iloc[:,1:]
            idx = int(df[df['SK_ID_CURR'] == loan.LoanID].index.values)
            break
    if idx == -1:
        raise "This loan id isn't in the database"
    del loan.LoanID, listSizeRow

    # chargement du model
    try: 
        model = pickle.load(bz2.open(join("dataviz","classifier.pkl"), "rb")) #abspath
    except:
        raise 'You must train the model first.'

    # préparation des objets pour l'explainer
    data_trans = model.steps[0][1].transform(df)
    data_trans = data_trans[idx].tolist()
    model_proba = model.steps[-1][1].predict_proba
    del df, idx

    # calcul des proba de prédiction
    log_proba = model.predict_proba(dataId).tolist()
    del model
    if log_proba[0][1] >= .3:
        predict = 0
    else:
        predict = 1
        
    try: 
        with gzip.open(join("dataviz","explainer"), "rb") as f:
            explainer = dill.load(f)
    except:
        raise 'You must download the interpretability object.'
        
    exp = explainer.explain_instance(np.array(data_trans)
                                    , model_proba
                                    , num_features = 20
                                    )
    res_exp = exp.as_list()
                                    
    del explainer, data_trans, model_proba, exp

    return {"log_proba":log_proba
            , "prediction":predict
            , "explainer":res_exp
            }
'''
@app.post('/explainer/')
async def get_explainer(data:list):
    # chargement du model
    try: 
        model = pickle.load(bz2.open(join("dataviz","classifier.pkl"), "rb")) #abspath
    except:
        raise 'You must train the model first.'
    model_proba = model.steps[-1][1].predict_proba
    del model

    return{'exp':exp.as_list()}
'''            
if __name__ == '__main__':
    uvicorn.run(app
                , host = "127.0.0.1" #'0.0.0.0'
                , port = 8000
                , debug = True # ligne à commenter lors du déploiement
               )