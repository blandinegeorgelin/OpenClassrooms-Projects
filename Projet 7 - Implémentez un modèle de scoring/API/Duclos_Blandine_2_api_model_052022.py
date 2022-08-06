# Library import
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import dill
import pandas as pd
from os.path import join, dirname, realpath

app = FastAPI(title = "Prediction Model API for Loan"
              , description = "A simple API that use machine learning model to predict the probability of difficulties paiements loan."
              # Can you predict how capable each applicant is of repaying a loan?
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
        model = pickle.load(open(join("dataviz","classifier.pkl"), "rb")) #abspath
    except:
        raise 'You must train the model first.'
    
    # features importances : already charge in the opening app
    try:
        feat_imp = model.steps[-1][1].feature_importances_
    except:
        feat_imp = model.steps[-1][1].coef_[0]
    del model

    df_feat_imp = pd.DataFrame(feat_imp
                                , columns = ["Coefficients"]
                                , index = model.feature_names_in_
                                )
    df_feat_imp = df_feat_imp.sort_values('Coefficients', ascending = False).head(20)
    return {"feat_imp": df_feat_imp}

@app.post('/predict/{id_loan}')
def predict_solvent(loan: Loan):
    if(not(loan)):
        raise "Please Provide a valid loan id"
    
    # chargement des données explicatives
    loanID = dict(loan)['LoanID']    
    try:
        df = pd.read_csv(join("dataviz",'data_test.csv'))
    except:
        raise 'No application train data is available, please check the csv file.'
    data = df[df['SK_ID_CURR'] == loanID]
    data = data.iloc[:,1:]
    if (data.shape[0] == 0):
        raise "This loan id isn't in the database"
    del df
    
    # chargement du model
    try: 
        model = pickle.load(open(join("dataviz","classifier.pkl"), "rb")) #abspath
    except:
        raise 'You must train the model first.'

    # préparation des objets pour explainer
    model_proba = model.steps[-1][1].predict_proba
    data_trans = model.steps[0][1].transform(data)

    # calcul des proba de prédiction
    log_proba = model.predict_proba(data).tolist()
    del model
    if log_proba[0][1] >= .3:
        predict = 0
        message_predict = "The data predict none of difficulties paiements. The loan is granted."
    else:
        predict = 1
        message_predict = "The data predict difficulties paiements. The loan is refused."
                         
    # Explainer
    n_feat = len(data_trans)
    try: 
        explainer = dill.load(open(join("dataviz","explainer"), "rb"))
    except:
        raise 'You must download the interpretability object.'
    exp = explainer.explain_instance(data_trans
                                    , model_proba
                                    , num_features = n_feat
                                    )
    del explainer

    return {"log_proba":log_proba
            , "prediction":predict
            , "message_predict":message_predict
            , "explainer":exp.as_html()
            }
            
if __name__ == '__main__':
    uvicorn.run(app
                , host = '0.0.0.0'
                #, port = 80
                #, debug = True # ligne à commenter lors du déploiement
               )