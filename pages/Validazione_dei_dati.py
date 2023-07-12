#Data analysis
import pdb
import pandas as pd
import os
import pickle 
import numpy as np
from datetime import datetime, date,timedelta
import streamlit as st
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import shap
from explainerdashboard import ClassifierExplainer,ExplainerDashboard, ExplainerHub
import streamlit.components.v1 as components

#General
from os import walk
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

#custom functions
from Funzioni_utili import bestclassifier,train,save_pickle,team_metrics,champions_metrics,d_in_future, confMatrix,\
download_excel,file_selector,doyourstupidthings, prediction, starting


###MAIN###
st.sidebar.markdown("# Validazione nel passato🎈")

with st.sidebar:
    st.write('Qui facciamo le validazioni. ')
st.title('Validazione nel passato')

st.subheader("""Cosa vorresti validare?""")
st.write('Il modello è allenato per il periodo 2012-2022 della championship del Brasile.')
#uploaded_file = st.file_uploader("Upload Excel to explore", type=".xlsx")
#path=os.getcwd()

#name = st.text_input("Nome e percorso del file", 'BRA_2012-2022hst_2023og_xVERO.xlsx')
year_col = st.text_input("Nome della colonna dell'anno", 'Season')
col_day = st.text_input("Nome della colonna della giornata", 'giornata')
start_year=st.text_input("Anno iniziale del dataset completo",2012)
start_year=int(start_year)
end_year=st.text_input("Anno finale del dataset completo",2022)
end_year=int(end_year)
anno_val=st.text_input("Anno su cui validare",2022)
anno_val=int(anno_val)
limite_prob=st.text_input("Probabilità oltre cui considerare pareggio",0.4)

#Campionato brasile. Un anno per ogni foglio
#Alleno su tutti gli anni, passando tutti gli anni sia per la media che per la predizione. 

anni=[*range(start_year,end_year+1)] #esclude 'l'ultimo anno
#valido sul 2022 che non ha mai visto.


#col_raw=['Giornata','Date','Time','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
col_raw=['Country','League','Season','Date','Time','Home','Away','HG','AG','Res']
st.write("Carica il modello e seleziona l'orizzonte previsionale")



uploaded_model = st.file_uploader("Carica modello. Limite: 200MB")


#'D_in_1iter', 'D_in_2iter', 'D_in_3iter',
outputs=['D_in_4iter','D_in_3iter','D_in_2iter','D_in_1iter']
uploaded_file = st.file_uploader("Carica excel", type=".xlsx")

if st.button('Prevedi for Braaasil',disabled=not(uploaded_file and uploaded_model), type='primary'):
    st.write(':leaves:')
    save=pickle.load(uploaded_model)
    output_select=save['Predict_future']
    st.write('Il modello prevede la probabilità che una squadra faccia almeno un pareggio nelle prossime {} giornate.'.format(output_select))

    output_choice=save['Output']

    [raw,final_df,int_df]=doyourstupidthings(uploaded_file,year_col,col_day,anni,anno_val,what='val')
    [int_df,alg_w]=prediction(save,output_choice,int_df)
    squadre=list(int_df.groupby(['SQUADRA']).mean().index)
    #val_df=int_df.copy()
    val_df=pd.DataFrame()
    for squadra in squadre:
        #Pari nelle prossime N partite da D=now a D=now+N
        temp=int_df[int_df['SQUADRA']==squadra]
        temp=temp.reset_index()
        (temp,outputs)=d_in_future(temp,4)
        val_df=pd.concat([val_df,temp])
    st.write('Ecco i dati completi')
    download_excel(val_df,name_exc='Prediction_Complete_{}'.format(anno_val))
    val_df=val_df.fillna(-1)
    st.write('Ordine: vero negativo, falso positivo, falso negativo, vero positivo')
    st.write('Confusion matrix')
    input=save['Input']
    cm = confusion_matrix(val_df[output_choice], val_df['{}_pred'.format(output_choice)])
    st.write(cm)

    probtarget='{}_probA'.format(output_choice)
    

    limite_prob=float(limite_prob)
    val_df[val_df[probtarget]>limite_prob]['{}_pred'.format(output_choice)]=1
    st.write('Confusion matrix con nuova probabilità')
    new_cm = confusion_matrix(val_df[output_choice], val_df['{}_pred'.format(output_choice)]))
    st.write(new_cm)


  
