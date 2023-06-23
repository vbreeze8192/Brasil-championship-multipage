#Data analysis
import os
import io
import pandas as pd
import pickle 
import numpy as np
from datetime import datetime, date,timedelta
import streamlit as st

#custom functions
from Funzioni_utili import bestclassifier,train,save_pickle,team_metrics,champions_metrics,d_in_future, confMatrix,\
download_excel,file_selector,doyourstupidthings,talk, prediction

###MAIN###
st.sidebar.markdown("# Previsioni per la champions del brasile🎈")

with st.sidebar:
    st.write('Qui facciamo le predizioni. ')
st.title('Brasile 2023')

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
anno_val=st.text_input("Anno in corso",2023)
anno_val=int(anno_val)
day=st.text_input('Giornata che si vuole predire',10)
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
    [raw,final_df,int_df]=doyourstupidthings(uploaded_file,year_col,col_day,anni,anno_val,day=day)
    [final_df,alg_w,alg_lp]=prediction(output_choice, final_df)
    final_df=talk(day,output_choice,final_df)

    st.write('Ecco i dati completi per la giornata {}.'.format(day))
    download_excel(final_df,name_exc='Prediction_Day{}'.format(day))
    st.balloons()

#######PROSSIMI SVILUPPI##########
## NEXT: PER I DATI CHE HO NEL PASSATO RENDERE DISPONIBILE VALIDAZIONE. XAI PER CAPIRE COME MAI. 
## SQUADRE CAMBIANO: 3 ALL'ANNO. 

##ANALISI
# QUALE DISTRIBUZIONE HA LA SINGOLA SQUADRA NEI PAREGGI QUEST'ANNO? QUANTO NEGLI ANNI PRECEDENTI?
# SOLO 3-4 ANNI NEL TRAINING
# PUNTEGGIO: DOVE SI TROVA LA SQUADRA RISPETTO ALLE ALTRE?
# PUNTEGGIO NEL PASSATO?
# QTY PAREGGI MASSIMI E MEDI!

####NUOVI INPUT
## INTEGRARE ANCHE I PUNTI DELLA SQUADRA GG PER GIORNO E LA MEDIA LA DEV STD E MAX MIN DI TUTTE LE ALTRE SQUADRE NELLA GIORNATA CORRENTE. 
## MEDIA E DEV STD DI PUNTI DELLA CHAMPIONSHIP E DI QUELLE PRECEDENTI
## E ANCHE MEDIA E DEV STD DEGLI ANNI PRECEDENTI DELLA STESSA SQUADRA.
## INFO SULLA DISTRIBUZIONE DEI PAREGGI NEL TEMPO. COME RAPPRESENTARLA? NON SOLO MAC DEI GIORNI MA MEDIA SU PARTITA E DEV STD SU PARTITA DI MAC NON PARI. 
## COPPIA DI SQUADRE: QUANDO PAREGGIANO INSIEME RENDILO. 1. INCROCIARE PROB PAREGGIO E MOSTRARE LE COPPIE PIù ALTE. 
## MODELLO CHE PREVEDE LA PROBABILITA' SHARP DI PAREGGIO AL GG 1, 2 3 O 4, NON LA SOMMA. 
## MODELLO PER COPPIE: CASA/FUORI 
## INTEGRARE MEDIA E DEV STD DELLA DISTRIBUZIONE DI GOAL NELLE ULTIME PARTITE PER SQUADRA PER COFRONTARE LE DISTRIBUZIONI 
## MODELLINO MULTICLASSE CHE PREVEDE TUTTI E QUATTRO. 