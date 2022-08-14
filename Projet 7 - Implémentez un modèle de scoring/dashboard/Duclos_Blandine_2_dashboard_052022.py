import streamlit as st
import pandas as pd
from os.path import join, dirname, realpath
import matplotlib.pyplot as plt
import requests
import pickle
import bz2
import gzip
import dill
import numpy as np

@st.cache(allow_output_mutation=True)
def load_data():
    with st.spinner('Calculating...'):
        # récupération de toutes les données clients
        try:
            df = pd.read_csv(join("data",'data_test.csv'), compression = 'bz2')
        except:
            raise 'No application data is available, please check the csv file.'
        idlist = ['< No id selecte >'] + df['SK_ID_CURR'].sort_values().to_list()

        # récupération des noms des colonnes suivant leur type
        catCol = []
        numCol = []
        for col in list(df.columns):
            if col != 'SK_ID_CURR':
                if df[col].nunique() <= 2:
                    catCol.append(col)
                else:
                    numCol.append(col)
    
    return df, idlist, catCol, numCol

@st.cache(allow_output_mutation=True)
def print_feat_imp():
    # graph of features importances model
    res = requests.get(f"https://blandine-duclos-api-model.herokuapp.com/features_importances")
    results = res.json()
    df_feat_imp = pd.DataFrame(results.get("feat_imp")).sort_values('Coefficients', ascending = False)
    fig, ax = plt.subplots()
    ax.barh(df_feat_imp.index
            , df_feat_imp.Coefficients
        )
    return fig

@st.cache(allow_output_mutation=True)
def print_predict(df, id_to_filter):
    # récupération des données du clients
    filtered_data = df[df['SK_ID_CURR'] == int(id_to_filter)]
    filtered_data = filtered_data.iloc[:,1:]

    # prédiction de la décision de l'accord ou non du prêt
    res = requests.post(f"https://blandine-duclos-api-model.herokuapp.com/predict/{id_to_filter}"
                        , json = {"LoanID": int(id_to_filter)}
                        )
    results = res.json()
    log_proba = results.get("log_proba")
    prediction = results.get("prediction")
    list_exp = results.get("explainer")
    if prediction == 0:
        message_predict = "loan granted"
    else:
        message_predict = "loan refused"
    return filtered_data, log_proba[0][1], prediction, message_predict, list_exp

@st.cache(allow_output_mutation=True)
def listVarUnique(catCol, numCol):
    listColUnique = ['Choose data to display'] + numCol
    for col in catCol:
        idx_last_col = col.rfind('_')
        name_col = col[:idx_last_col]
        if name_col not in listColUnique:
            listColUnique.append(name_col)

    return listColUnique

@st.cache(allow_output_mutation=True)
def list_dataCat(df, id_to_filter, varcat):
    dataCust = df[df['SK_ID_CURR'] != id_to_filter]

    listVar = [x for x in df.columns.tolist() if x.startswith(varcat)]
    dataCust = dataCust.loc[:,listVar]
    valueCust = dataCust.idxmax(axis = 1).values.tolist()[0]

    dfCatCol = pd.DataFrame(columns = ['Label','Value'])
    for col in listVar:
        idx_last_col = col.rfind('_')
        name_value = col[idx_last_col+1:]
        if col == valueCust:
            valueCust = name_value
        values = df.loc[df['SK_ID_CURR'] != id_to_filter, col].sum()
        dict_insert = {'Label':name_value
                        , 'Value':values}
        dfCatCol = pd.concat([dfCatCol,pd.DataFrame([dict_insert])], ignore_index = False).reset_index(drop = True)

    return valueCust, dfCatCol

@st.cache(allow_output_mutation=True)
def print_catcol(df, id_to_filter, varcat):
    valueCust, dfCatCol = list_dataCat(df, id_to_filter, varcat)

    labelList = dfCatCol['Label']
    valueList = dfCatCol['Value']
    fig, ax = plt.subplots()
    ax.bar(labelList
            , valueList
            )
    xi = valueCust
    yi = dfCatCol.loc[dfCatCol['Label'] == xi,'Value'] / 2
    ax.scatter(xi, yi)
    
    return fig

@st.cache(allow_output_mutation=True)
def print_numcol(dfCatCol, id_to_filter, varNum):
    data = df.loc[df['SK_ID_CURR'] != id_to_filter, varNum]
    fig, ax = plt.subplots()
    ax.boxplot(data, showfliers = False)
    ax.scatter(varNum
                , filtered_data[varNum]
                , c = 'red')
    return fig

@st.cache(allow_output_mutation=True)
def print_shap(list_exp):
    '''
    res = requests.post(f"https://blandine-duclos-api-model.herokuapp.com/explainer/"
                        , json = {"data": data_exp}
                        )
    results = res.json()
    '''
    plot = pd.DataFrame(list_exp, columns = ['Label','Value'])
    color = ['r' if y < 0 else 'g' for y in plot['Value']]
    fig, ax = plt.subplots()
    ax.barh(plot['Label']
            , plot['Value']
            , color = color
        )
    
    return fig

st.title('My Prediction App 🎯')
st.write("""
        This is an application to know if a loan are granted to a customer and what influenced this decision.
        """)

st.sidebar.write("""
        ## Filters
        """)
df, idlist, catCol, numCol = load_data()

# Selected ID Customer
id_to_filter = st.sidebar.selectbox('Select the customer id :'
                                    , idlist
                                    )

selected_global_feat_imp = st.sidebar.checkbox('Global importance of data in the decision loan'
                                                , value = True)

# Affichage de l'importance globale des données clients dans les prédictions de l'accord ou non du prêt
if selected_global_feat_imp:
    st.subheader(f"Global importance of data in the decision loan")
    with st.spinner('Calculating...'):
        fig = print_feat_imp()
        st.pyplot(fig)

if id_to_filter != '< No id selecte >':
    # affichage des options liés aux données clients
    selected_data = st.sidebar.checkbox("Customer's data")
    selected_shape_values = st.sidebar.checkbox('Importance of data in the decision loan')
    with st.spinner('Calculating...'):
        listVar = listVarUnique(catCol, numCol)
        selected_var = st.sidebar.selectbox('Select the data to display :'
                                            , listVar
                                            )

    with st.spinner('Calculating...'):
        filtered_data, log_proba, prediction, message_predict, list_exp = print_predict(df, id_to_filter)

    st.subheader(f"Loan decision for the customer : {id_to_filter}")
    st.write(f"""
            Probability of difficulties paiements : {round(log_proba,4)}
            """)
    st.write(f"""
            Status Loan {prediction} : {message_predict}
            """)

    # Affichage des données descriptives du customer
    if selected_data: 
        st.subheader(f"Customer's data for the id : {id_to_filter}")
        with st.spinner('Calculating...'):
            st.dataframe(filtered_data)

    # Affichage des graphiques de comparaison Customer VS CustomerS
    if selected_var != 'Choose data to display':

        st.subheader(f"Comparison between the data of client ({id_to_filter}) and the others")
        
        with st.spinner('Calculating...'):
            if selected_var in numCol:
                # point plot for all other col
                    fig = print_numcol(df, id_to_filter, selected_var)
            else:
                # barplot for all catCol
                    fig = print_catcol(df, id_to_filter, selected_var)
            st.pyplot(fig)

    # Affichage des données expliquant la décision du prêt
    if selected_shape_values: 
        st.subheader(f"Importance of data in the loan decision for the customer : {id_to_filter}")

        with st.spinner('Calculating...'):
            fig = print_shap(list_exp)
            st.pyplot(fig)