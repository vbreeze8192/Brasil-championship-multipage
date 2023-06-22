#Data analysis
import pdb
import pandas as pd
import os
import pickle 
import numpy as np
from datetime import datetime, date,timedelta
import streamlit as st

#General
from os import walk
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

#custom functions
from Funzioni_utili import bestclassifier,train,save_pickle,team_metrics,champions_metrics,d_in_future, confMatrix,\
download_excel,file_selector,doyourstupidthings


###MAIN###
st.sidebar.markdown("# Previsioni per la champions del brasile🎈")

with st.sidebar:
    st.write('Qui facciamo le predizioni. ')
st.title('Validazione nel passato')

st.subheader("""Cosa vorresti prevedere?""")
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
day=st.text_input('Giornata che si vuole validare',10)
day=int(day)
#Campionato brasile. Un anno per ogni foglio
#Alleno su tutti gli anni, passando tutti gli anni sia per la media che per la predizione. 

anni=[*range(start_year,end_year+1)] #esclude 'l'ultimo anno
#valido sul 2022 che non ha mai visto.


#col_raw=['Giornata','Date','Time','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
col_raw=['Country','League','Season','Date','Time','Home','Away','HG','AG','Res']

output_select = st.radio(
    "Su quanti giorni vuoi prevedere la cumulata dei pareggi?",
    ('1','3','4'))

output_choice = 'D_in_{}iter'.format(output_select)
#'D_in_1iter', 'D_in_2iter', 'D_in_3iter',
st.write('Il modello prevede la probabilità che una squadra faccia almeno un pareggio nelle prossime {} giornate.'.format(output_select))
outputs=['D_in_4iter','D_in_2iter','D_in_1iter']
uploaded_file = st.file_uploader("Carica excel", type=".xlsx")

if st.button('Prevedi for Braaasil',disabled=not uploaded_file, type='primary'):
    st.write(':leaves:')
    val_df=pd.DataFrame()
    [day_iter,output_choice,final_df]=doyourstupidthings(uploaded_file,year_col,col_day,anni,anno_val,output_choice,day)
    squadre=list(final_df.groupby(['SQUADRA']).mean().index)
    for squadra in squadre:
        #Pari nelle prossime N partite da D=now a D=now+N
        temp=final_df[final_df['SQUADRA']==squadra]
        (temp,outputs)=d_in_future(temp,4)
        val_df=pd.concat([val_df,temp])
    st.write('Ecco i dati completi per la giornata {}.'.format(day))
    download_excel(val_df,name_exc='Prediction_Day{}'.format(day))
  