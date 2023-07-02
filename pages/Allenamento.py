#Data analysis
import pdb
import pandas as pd
import os
import pickle 
import numpy as np
from datetime import datetime, date,timedelta
import streamlit as st
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

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
st.sidebar.markdown("# Allenamento dei modelli🎈")

with st.sidebar:
    st.write("Qui facciamo l'allenamento.")
st.title('Allenamento')

st.subheader("""Struttura il dataset di training""")
st.write('Scegli in periodo di training e validazione.')
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

#Campionato brasile. Un anno per ogni foglio
#Alleno su tutti gli anni, passando tutti gli anni sia per la media che per la predizione. 

anni=[*range(start_year,end_year+1)] #esclude 'l'ultimo anno
#valido sul 2022 che non ha mai visto.


#col_raw=['Giornata','Date','Time','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
col_raw=['Country','League','Season','Date','Time','Home','Away','HG','AG','Res']

#'D_in_1iter', 'D_in_2iter', 'D_in_3iter',
st.write('Il modello prevede la probabilità che una squadra faccia almeno un pareggio nelle prossime 1, 2, 3 e 4 giornate.')
outputs=['D_in_4iter','D_in_3iter','D_in_2iter','D_in_1iter']
uploaded_file = st.file_uploader("Carica excel", type=".xlsx")

if st.button('Allena for Braaasil',disabled=not uploaded_file, type='primary'):
    st.write(':leaves:')
    
    [raw,final_df,int_df]=doyourstupidthings(uploaded_file,year_col,col_day,anni,anno_val,what='train')

    squadre=list(int_df.groupby(['SQUADRA']).mean().index)
    train_df=int_df.copy()
    for squadra in squadre:
        #Pari nelle prossime N partite da D=now a D=now+N
        temp=int_df[int_df['SQUADRA']==squadra]
        (temp,outputs)=d_in_future(temp,4)
        train_df=pd.concat([train_df,temp])

    for output in outputs:
        st.write('Sto calcolando questo: {}'.format(output))
        [input,input_lower]=starting()
        st.write("\n:robot_face: E mo' m'annleno. :robot_face:")
        train_df=train_df.fillna(0)
        download_excel(train_df,name_exc='Training')
        [alg,dicts,nome_modello]=train(train_df,input,output,task='rfc',testsize=0.3,nome_modello='{}_model_v02'.format(output))
        #final_df['{}_prob'.format(output)]=alg.predict_proba(final_df[input])
        
        
        st.write('Confusion matrix su tutto il dataset')
        st.write('Ordine: vero negativo, falso positivo, falso negativo, vero positivo')
        cm = confusion_matrix(train_df[output], alg.predict(train_df[input]))
        st.write(cm)
        st.download_button(
            "Download Model",
            data=pickle.dumps(alg),
            file_name='{}_model_v02',
        )
    

 
    





  
