# Library import
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import dill
import pandas as pd
from os.path import join, dirname, realpath

# load training model
try: 
    model = pickle.load(open(join(dirname(realpath(__file__)),"dataviz","classifier.pkl"), "rb")) #abspath
except:
    raise 'You must train the model first.'
    
# load application_test data
try:
    df = pd.read_csv(join(dirname(realpath(__file__)),"dataviz",'data_test.csv'))
except:
    raise 'No application train data is available, please check the csv file.'

# Load interpretability model
try: 
    explainer = dill.load(open(join(dirname(realpath(__file__)),"dataviz","explainer"), "rb"))
except:
    raise 'You must download the interpretability object.'

# Create app
app = FastAPI(title = "Prediction Model API for Loan"
              , description = "A simple API that use machine learning model to predict the probability of difficulties paiements loan."
              # Can you predict how capable each applicant is of repaying a loan?
              , version = "0.1",)

# Class which describes a single loan
class Loan(BaseModel):
    LoanID: int
    
# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome on the Loan Classification App"}
    
# Expose the features importances globals
@app.get("/features_importances")
def features_importances_globals():
    # features importances : already charge in the opening app
    try:
        feat_imp = model.steps[-1][1].feature_importances_
    except:
        feat_imp = model.steps[-1][1].coef_[0]
    df_feat_imp = pd.DataFrame(feat_imp
                                , columns = ["Coefficients"]
                                , index = model.feature_names_in_
                                )
    df_feat_imp = df_feat_imp.sort_values('Coefficients', ascending = False).head(20)
    return {"feat_imp": df_feat_imp}

# Expose the prediction functionality, make a probability prediction and the model interpretability
@app.post('/predict/{id_loan}')
def predict_solvent(loan: Loan):
    if(not(loan)):
        raise "Please Provide a valid loan id"
    
    # Résultats de décision
    loanID = dict(loan)['LoanID']
    data = df[df['SK_ID_CURR'] == loanID]
    data = data.iloc[:,1:]
    if (data.shape[0] == 0):
        raise "This loan id isn't in the database"

    log_proba = model.predict_proba(data).tolist()
    if log_proba[0][1] >= .3:
        predict = 0
        message_predict = "The data predict none of difficulties paiements. The loan is granted."
    else:
        predict = 1
        message_predict = "The data predict difficulties paiements. The loan is refused."
                         
    # Explainer
    idx = df[df['SK_ID_CURR'] == loanID].index.tolist()[0]
    data_trans = model.steps[0][1].transform(df)
    data_trans = data_trans[idx]
    model_proba = model.steps[-1][1].predict_proba
    n_feat = len(data_trans)
    exp = explainer.explain_instance(data_trans
                                    , model_proba
                                    , num_features = n_feat
                                    )

    return {"log_proba":log_proba
            , "prediction":predict
            , "message_predict":message_predict
            , "explainer":exp.as_html()
            }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app
                , host = '0.0.0.0'
                #, port = 80
                #, debug = True # ligne à commenter lors du déploiement
               )
    
# on Terminal : 
#    python -m uvicorn name_file:name_app_in_file --reload
#    python -m uvicorn Duclos_Blandine_2_api_model_052022:app --reload
# Le --reload indique que vous souhaitez que l'API s'actualise automatiquement lorsque vous enregistrez le fichier sans redémarrer l'ensemble.

# on Browser : (only for get test)
#    http://127.0.0.1:8000/docs
# documentation interactive intégrée
#    http://127.0.0.1:8000/redoc
# autre style de documentation interactive intégrée

# on Postman : (only for post test)
#    https://www.getpostman.com
