import streamlit as st
import pandas as pd
from os.path import join, dirname, realpath
import matplotlib.pyplot as plt
import requests

try:
    df = pd.read_csv(join("data",'data_test.csv'))
except:
    raise 'No application train data is available, please check the csv file.'

st.sidebar.write("""
        ## Filters
        Select the filter that you want and wait
        """)

# Selected ID Customer
list_custId = ['< No id selecte >']
for id in df['SK_ID_CURR'].sort_values().to_list():
    list_custId.append(id)
id_to_filter = st.sidebar.selectbox('Select the customer id :'
                    , list_custId
                    )

catCol = []
numCol = []
for col in df.columns.to_list():
    if col != 'SK_ID_CURR':
        if df[col].nunique() <= 2:
            catCol.append(col)
        else:
            numCol.append(col)

st.title('My Prediction App 🎯')
st.write("""
        This is an application to know if a loan are granted to a customer and what influenced this decision.
        """)

# Selected Features importances globals
selected_global_feat_imp = st.sidebar.checkbox('Global importance of data in the decision loan'
                                                , value = True)
if selected_global_feat_imp:
    st.subheader(f"Global importance of data in the decision loan")
    # graph of features importances model
    res = requests.get(f"https://blandine-duclos-api-model.herokuapp.com/features_importances")
    results = res.json()
    df_feat_imp = pd.DataFrame(results.get("feat_imp"))
    fig, ax = plt.subplots()
    ax.barh(df_feat_imp.index
            , df_feat_imp.Coefficients
        )
    st.pyplot(fig)

if id_to_filter != '< No id selecte >':
    filtered_data = df[df['SK_ID_CURR'] == int(id_to_filter)]
    filtered_data = filtered_data.iloc[:,1:]
    
    selected_data = st.sidebar.checkbox("Customer's data")
    with st.spinner('Calculating...'):
        id_loan = {"loan": str(id_to_filter)}
        res = requests.post(f"https://blandine-duclos-api-model.herokuapp.com/predict/{id_to_filter}"
                            , json = {"LoanID": int(id_to_filter)}
                            )
        results = res.json()
        log_proba = results.get("log_proba")
        prediction = results.get("prediction")
        message_predict = results.get("message_predict")
        explainer = results.get("explainer")
        st.subheader(f"Loan decision for the customer : {id_to_filter}")
        st.write(f"""
                Probability of difficulties paiements : {round(log_proba[0][1],4)}
                """)
        st.write(f"""
                Status Loan : {prediction}\n
                {message_predict}
                """)

    if selected_data: # Affichage des données descriptives du customer
        with st.spinner('Calculating...'):
            st.subheader(f"Customer's data for the id : {id_to_filter}")
            st.dataframe(filtered_data)
    
    st.sidebar.write("""
                    ## Graphs to display
                    """)
    selected_comp_cust = st.sidebar.checkbox("Comparison between the customer's data and that of others")
    selected_shape_values = st.sidebar.checkbox('Importance of data in the decision loan')
    
    if selected_comp_cust: # Affichage des graphiques de comparaison Customer VS CustomerS
        with st.spinner('Calculating...'):
            st.subheader(f"Comparison between customer's data ({id_to_filter}) and that of other")
            dataCust = df[df['SK_ID_CURR'] != id_to_filter]
            dataCust = df.iloc[:,1:]

            dfCatCol = pd.DataFrame(columns = ['Col','Label','Value'])
            for col in catCol:
                idx_last_col = col.rfind('_')
                name_col = col[:idx_last_col]
                name_value = col[idx_last_col+1:]
                value = df.loc[df['SK_ID_CURR'] != id_to_filter, col].sum()
                dict_insert = {'Col':name_col
                                , 'Label':name_value
                                , 'Value':value}
                dfCatCol = pd.concat([dfCatCol,pd.DataFrame([dict_insert])], ignore_index = False).reset_index(drop = True)

            listVarCat = dfCatCol['Col'].unique().tolist()
            listVar = numCol + listVarCat
            selected_var = st.sidebar.selectbox('Select the data to display :'
                                                    , listVar
                                                    )
            with st.spinner('Calculating...'):
                if selected_var in listVarCat:
                    # barplot for all catCol
                    fig, ax = plt.subplots()
                    labelList = dfCatCol.loc[dfCatCol['Col'] == selected_var,'Label'].unique().tolist()
                    valueList = dfCatCol.loc[dfCatCol['Col'] == selected_var,'Value']
                    ax.bar(labelList
                            , valueList
                        )
                    i = 0
                    j = 0
                    while i == 0:
                        if filtered_data[selected_var+'_'+labelList[j]].values[0] == 1:
                            xi = labelList[j]
                            i = 1
                    yi = dfCatCol.loc[(dfCatCol['Col'] == selected_var)&(dfCatCol['Label'] == xi),'Value'] / 2
                    ax.scatter(xi, yi)
                    st.pyplot(fig)
                else:
                    # point plot for all other col
                    data = df.loc[df['SK_ID_CURR'] != id_to_filter, selected_var]
                    fig, ax = plt.subplots()
                    ax.boxplot(data, showfliers = False)
                    ax.scatter(1
                                , filtered_data[selected_var]
                                , c = 'red')
                    st.pyplot(fig)

    if selected_shape_values: # Affichage des données expliquant la décision du prêt
        with st.spinner('Calculating...'):
            st.subheader(f"Importance of data in the loan decision for the customer : {id_to_filter}")
            st.components.v1.html(explainer)