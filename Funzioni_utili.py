#Data analysis
import pandas as pd
import os
import pickle 
import numpy as np
import streamlit as st
import io 

#Modeling
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.pipeline import Pipeline
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import r2_score


def starting(print_input='NO'):
    
    dict_input={'AVG_D_3Y_CH':'Media di pareggi per championship, calcolata sulle stagioni precedenti',\
    'AVG_D_N_CH':'Media di pareggi per championship, calcolata sulla stagione attuale',\
    'AVG_ND_3Y_S':'Media di non-pareggi per squadra, calcolata sulle stagioni precedenti',\
    'AVG_ND_N_S':'Media di non-pareggi per squadra, calcolata sulla stagione attuale',\
    'AVG_Dxd_3Y_CH':'Media di pareggi per giornata per championship, calcolata sulle stagioni passate',\
    'AVG_Dxd_N_CH':'Media di pareggi per giornata per championship, calcolata sulla stagione presente',\
    'QTY_ND_3Y_S':'Media del periodo massimo per stagione di giornate consecutive senza pareggi per squadra, calcolata sulle stagioni precedenti',\
    'QTY_ND_N_S':'Quantità di giornate consecutive senza pareggi per squadra sulla stagione attuale',\
    'MEAN_S':'Media dei goal per partita nella stagione attuale della squadra di riferimento',\
    'STD_S':'Deviazione standard dei goal per partita nella stagione attuale della squadra di riferimento',\
    'MEAN_S_SFID':'Media dei goal per partita nella stagione attuale della squadra sfidante',\
    'STD_S_SFID':'Deviazione standard dei goal per partita nella stagione attuale della squadra sfidante',\
    
    'AVG_ND_3Y_S_SFID':'Media di non-pareggi per sfidante, calcolata sulle stagioni precedenti',\
    'AVG_ND_N_S_SFID':'Media di non-pareggi per sfidante, calcolata sulla stagione attuale',\
    'QTY_ND_3Y_S_SFID':'Media del periodo massimo per stagione di giornate consecutive senza pareggi per sfidante, calcolata sulle stagioni precedenti',\
    'QTY_ND_N_S_SFID':'Quantità di giornate consecutive senza pareggi per sfidante sulla stagione attuale',\
    
    'HOUR':'Ora della partita',\
    'HoA':'Indicazione su Home o Away (0: Away, 1: Home)'}

    if print_input!='NO':
         for item in dict_input:
              st.write(':red[{}: ]       {}'.format(item,dict_input[item]))
                       

    input_lower=['AVG_ND_3Y_S',\
    'AVG_ND_N_S',\
    'QTY_ND_3Y_S',\
    'QTY_ND_N_S',\
    'HOUR',\
    'HoA']
    '''
    input=['AVG_D_3Y_CH',\
    'AVG_D_N_CH',\
    'AVG_ND_3Y_S',\
    'AVG_ND_N_S',\
    'AVG_Dxd_3Y_CH',\
    'AVG_Dxd_N_CH',\
    'QTY_ND_3Y_S',\
    'QTY_ND_N_S',\
    'MEAN_S',\
    'STD_S',\
    'MEAN_S_SFID',\
    'STD_S_SFID',\
    'HOUR',\
    'HoA']
    '''
    input=[]
    for item in dict_input:
         input=input+[item]

    return(input,input_lower)
     
#Funzioni per allenare il modello 
def rfc(X_train, y_train,X_test,y_test):
    #np.random.randint(50,high=500)
    param_dist = {'n_estimators': [20,50,100,200,300,400,500,600],
            'max_depth': [10,30,50,60,100,150,200,250], 'min_samples_split':[2, 3, 5, 7,9,12],
                'criterion':['gini', 'entropy', 'log_loss'], 'min_samples_leaf':[1,2,3],'class_weight':['balanced', 'balanced_subsample','None']}

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                param_distributions = param_dist, 
                                n_iter=5, 
                                cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)

    
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    st.write('Best hyperparameters:',  rand_search.best_params_)
    # Generate predictions with the best model
    y_pred = best_rf.predict(X_test)
    st.write('Score:',rand_search.score(X_test, y_test))

    st.write(classification_report(y_test, y_pred, output_dict=True))
    #confMatrix(y_test, y_pred)
    st.write('Confusion matrix su test set')
    cm = confusion_matrix(y_test,best_rf.predict(X_test))
    st.write(cm)


    return(best_rf,rand_search.score(X_test, y_test))

def confMatrix(y_test,y_pred,title='Confusion Matrix'):
          # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp =ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    disp.ax_.set_title(title)

def bestclassifier(X_train, y_train,X_test,y_test,gridsearch=True):

    """Finds the best pipeline for the data provided within classifier, using gridsearch or randomizedsearch""" 
    # Initialize the estimators
    clf1=ExtraTreesClassifier(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    #clf3=XGBClassifier(random_state=42,gamma=0,scale_pos_weight=1,validation_fraction=0.1)
    
    #Algorithm settings
    settings1={'classifier__n_estimators':[50,200,350,500],'classifier__max_depth':[3,4,5,6], 'classifier__min_samples_split':[3, 5, 7],'classifier':[clf1]}
    settings2 = {'classifier__n_estimators':[40,50,70,80,100], 'classifier__max_depth':[3,4], 'classifier__min_samples_split':[3, 5, 7],'classifier':[clf2]}
    #settings3 = {'classifier__max_depth':[2,3,5], 'classifier__min_child_weight':[2,3,5],'classifier__n_estimators':[50,100,150],\
     #   'classifier__learning_rate':[0.5,0.2,0.1,0.05],'classifier':[clf3]}

    #Final pipeline
    params = [settings1, settings2]#, settings3]
    pipe = Pipeline([\
        ('scl', StandardScaler()),\
        ('classifier', clf1)])

    #Model search
    if gridsearch==True:
        #With gridsearch:
        gs=GridSearchCV(pipe, params, cv=3, n_jobs=-1,verbose=-1).fit(X_train, y_train) 
    else:
        #With Random search:
        gs=RandomizedSearchCV(pipe, params, cv=3, n_jobs=-1, verbose=-1).fit(X_train, y_train)
    
    print('Found best estimator')
    my_params = gs.best_params_
    print('Best algorithm:\n', gs.best_params_['classifier'])
    print('Best params:\n', gs.best_params_)
    alg=gs.best_estimator_
    prediction = gs.predict(X_test)
    for ii in range(0,10):
        print('P:{}, R:{}'.format(prediction[ii],y_test[ii]))
    
    # Create confusion matrix
    print('Confusion matrix: {}'.format(confusion_matrix(y_test, prediction)))
    # Display accuracy score
    print('Accuracy score: {}'.format(accuracy_score(y_test, prediction)))
    # Display F1 score
    #print('F1 Score: {}'.format(f1_score(y_test,prediction)))
    

    print('GridSearchCV accuracy:\n', gs.score(X_test, y_test))
    return(alg,gs.score(X_test, y_test))



    


def train(df,inputs, OUTPUT,task='reg',testsize=0.25,path=os.getcwd(),nome_modello='NA'):
    """Uses best model to train a model"""
    #TODO: define what's test, validation, train
    ordir=os.getcwd()
    df_train,df_test = train_test_split(df,test_size=testsize)

    X_train = df_train[inputs]
    y_train = df_train[OUTPUT]
    X_test = df_test[inputs]
    y_test = df_test[OUTPUT]
    output_mean = y_test.mean()

    if task=='reg':
        [alg,score_alg]=bestregressor(X_train, y_train,X_test, y_test)
    elif task=='clf':
        [alg,score_alg]=bestclassifier(X_train, y_train,X_test, y_test)
    elif task=='rfc':
        [alg,score_alg]=rfc(X_train, y_train,X_test, y_test)
    else:
        raise Exception('Task must be reg for regression and clf for classification. rfc for random foresst classifier.')
    os.chdir(path)
    (dicts,nome_modello)=save_pickle(alg,inputs,OUTPUT,\
    score_alg)
    os.chdir(ordir)
    return(alg,dicts,nome_modello)

def save_pickle(alg,inlist,outlist,score,nome_modello='NA'):
    """Saves pickle in quasi-standard structure for Rebecca"""
    if nome_modello=='NA':
        nome_modello='Modello_{}'.format(outlist)
    save={}
    save['Algorithm']=alg
    save['Input']=inlist
    save['Output']=outlist
    save['Score_test']=score
    pickle.dump(save, open(nome_modello, 'wb'))
    pickle.dump(save['Algorithm'], open('{}_concrete_algorithm'.format(nome_modello), 'wb'))
    print('Saved')
    return(save,nome_modello)
     
#funzioni per estrarre le info da utilizzare in train e test?
#indicatori da champion 3 yr prec: da calcolare sul raw o sul primo livello di tabella? Rischi oche si raddoppi? non dovrebbe sulle medie
def team_metrics(df,squadre,col_squadre='SQUADRA',col_d='D'):
        '''Calcola le metriche per campionati precedenti'''

        #Media di non pari nel periodo considerato per squadra: df con index=squadra, value=media
        nd_s=df.groupby([col_squadre]).mean()[col_d] 

        #Quantity di non pari longer nel periodo considerato: df con index=squadra, value=media
        qtymax_s=pd.DataFrame(index=squadre,columns=['qtymax_s'])

        ii=0
        
        for squadra in squadre:
                #Itero su tutte le squadre per creare il df
                maxx=0
                temp=df[df[col_squadre]==squadra]
                #itero su tutte le osservazioni per trovare il massimo di d consecutivi
                for item in range(0,len(temp)):
                        if temp[col_d].iloc[item]==0:
                                ii=ii+1
                                if ii>maxx:
                                        maxx=ii
                                        ii=0
                        
                qtymax_s.loc[squadra]=maxx
        
        return(nd_s,qtymax_s)

def champions_metrics(df,col_d='D',col_day='Giornata'):
        '''Calcola le metriche per campionati precedenti'''
        #Media di pari nel periodo considerato per campionato: valore singolo
        avg_ch=df[col_d].mean() #da capire se ok

        #Media di pari per giornata nel periodo considerato per campionato: valore singolo
        avgdxd_ch=df.groupby([col_day]).mean().mean()[col_d] #primo mean df con medie per ogni giorno, secondo mean media su tutti i giorni

        
        return(avg_ch,avgdxd_ch)


def d_in_future(df,Nmax,col_d='D'):
        out_list=[]
        for N in range(1,Nmax+1):
                out_list=out_list+['{}_in_{}iter'.format(col_d,N)]
                df['{}_in_{}iter'.format(col_d,N)]=0
                for ii in range(0,len(df.index)-N):
                        start=df.index[ii]
                        stop=df.index[ii+N] #da capire se viene nelle successive 3 o 4
                        if df[col_d].loc[start:stop].sum()>0:
                                df['{}_in_{}iter'.format(col_d,N)].loc[start]=1
                df['{}_in_{}iter'.format(col_d,N)]=df['{}_in_{}iter'.format(col_d,N)].astype(int)	
        #eventualmente aggiungere colonna aggiuntiva per numero di partita di x
        return(df,out_list)

#
def download_excel(dftoexc,name_exc='Download_Excel'):
    # buffer to use for excel writer
    buffer = io.BytesIO()
    st.write(dftoexc)
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        dftoexc.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.save()
    st.download_button(
    label="Download data as Excel",
    data=buffer,
    file_name='{}.xlsx'.format(name_exc),
    mime='application/vnd.ms-excel')
        
    
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)



def doyourstupidthings(name,year_col,col_day,anni,anno_val,day='NA',what='pred',col_date='Date',col_time='Time',col_a='Away',col_ag='AG',col_h='Home',col_hg='HG',col_res='Res'):
    [input,input_lower]=starting()

    original=pd.DataFrame()
    for anno in anni+[anno_val]:
        anno=str(anno)
        temp=pd.read_excel(name,sheet_name=anno)
        original=pd.concat([original,temp])


    st.write(':crown: Dati importati, bitches! :crown:')
    #crea colonna D con i pareggi
    original['D']=original[col_res].copy()
    original['D'].iloc[original[col_res]=='D']=1
    original['D'].iloc[original[col_res]!='D']=0
    #print(original)

    #dividi le squadre: crea 2 dataset temporanei che contengono solo le singole squadre home e away e relativo risultato
    temp_a=original[[col_day,year_col,col_date,col_time,col_a,col_h,col_hg,col_ag,'D']]
    temp_h=original[[col_day,year_col,col_date,col_time,col_h,col_a,col_ag,col_hg,'D']]

    #cambia nome delle colonne 
    temp_a=temp_a.rename(columns={col_a: "SQUADRA",col_h:"SFIDANTE",col_ag: "N_GOAL",col_hg: "N_GOAL_SF"})
    temp_h=temp_h.rename(columns={col_h: "SQUADRA",col_a:"SFIDANTE", col_hg: "N_GOAL", col_ag: "N_GOAL_SF"})
    temp_a['HoA']='0'
    temp_h['HoA']='1'

    #UNISCI I DUE DF
    raw=pd.concat([temp_h, temp_a])
    st.write(':floppy_disk: Ho fatto il dataset completo. Ora estraggo i dati utili: squadre, pareggi, altre amenità.')
    #estrai lista delle squadre
    squadre=list(raw.groupby(['SQUADRA']).mean().index)

    raw['HOUR']=0
    tl=st.empty()
    for ii in range(0,len(raw)):
        try:
            raw['HOUR'].iloc[ii]=int(raw[col_time].iloc[ii].hour)
        except Exception as e:
            with tl.container():
                st.write(':red[Found error: {}, iter {}]'.format(e,ii))
             
    '''
    for col in [col_hg,col_ag,col_res]:
        raw[col]=0
    '''

    #st.write('Valori non validi: {}'.format(raw.isna().sum()))
    raw=raw.fillna(0)

    for col in [col_day, 'N_GOAL','N_GOAL_SF', 'D', 'HoA', year_col]:
        raw[col]=raw[col].astype(int)

    st.write(':calendar: Divido il dataset completo in anni prima e anno corrente.')

    df3yr=pd.DataFrame()
    for anno in anni:
        df3yr=pd.concat([df3yr,raw[raw[year_col]==anno]])
    st.write(':white_check_mark: Nel passato ci sono queste squadre:')
    st.write(list(df3yr.groupby(['SQUADRA']).mean().index))

    st.write('Righe degli anni prima: {}'.format(df3yr.shape[0]))

    #Metriche di campionato
    st.write(':trophy: Calcolo le metriche per la championship e per singola squadra.')
    [avg3yrch,avgdxd3yrch]=champions_metrics(df3yr,col_day=col_day)
    #[nd3yrs,qtymax3yrs]=team_metrics(df3yr,squadre) 
    qty=pd.DataFrame(index=anni)
    ndr=pd.DataFrame(index=anni)

    qty_s=pd.DataFrame(index=anni)
    ndr_s=pd.DataFrame(index=anni)

    for anno in anni:
        #SQUADRA
        temp=df3yr[df3yr[year_col]==anno]
        [ndt,qtyt]=team_metrics(temp,squadre)

        ndt=ndt.reset_index()
        qtyt=qtyt.reset_index()
        qtyt=qtyt.rename(columns={"index": "SQUADRA"})
        qty=pd.concat([qty,qtyt]) #concateno anno su anno
        ndr=pd.concat([ndr,ndt])

        #SFIDANTE
        temp=df3yr[df3yr[year_col]==anno]
        [ndt_s,qtyt_s]=team_metrics(temp,squadre,col_squadre='SFIDANTE')

        ndt_s=ndt_s.reset_index()
        qtyt_s=qtyt_s.reset_index()
        qtyt=qtyt.rename(columns={"index": "SQUADRA"})
        qty_s=pd.concat([qty,qtyt]) #concateno anno su anno
        ndr_s=pd.concat([ndr,ndt])

    nd3yrs=ndr.groupby('SQUADRA').mean() #questo dovrebbe fare la media per squadra
    qtymax3yrs=qty.groupby('SQUADRA').mean()
    nd3yrs_sf=ndr_s.groupby('SQUADRA').mean()
    qtymax3yrs_sf=qty_s.groupby('SQUADRA').mean()

    #raw è la rappresentazione dell'anno in corso. 
    #sul singolo anno di validazione, valuta sia i gol nel futuro veri, sia la predizione fatta dal modello. 
    raw_complete=raw.copy()
    raw=raw[raw[year_col]==anno_val]
    st.write('Righe per questo anno: {}'.format(raw.shape[0]))
    raw=raw.sort_values(col_day)

    
    df=pd.DataFrame()
    st.write('	:hourglass_flowing_sand: Guardo i dati per giornata...')
    if what=='pred':
        anni_iter=[anno_val]
        if day=='NA':
            day=raw[col_day].iloc[-1]
        nn=day+1
    elif what=='val':
            anni_iter=[anno_val]
            nn=raw[col_day].iloc[-1]
            if day=='NA':
                day=5
    elif what=='train':
            nn=raw[col_day].iloc[-1]
            day=5
            anni_iter=anni
    else:
        st.write('Devi specificare se pred o val o train. Nel dubbio predico.')
        if day=='NA':
            day=raw[col_day].iloc[-1]
        nn=day+1         
        #in che giorno nella riga

    
    int_df=pd.DataFrame()
    final_df=pd.DataFrame()
    for anno in anni_iter:
        raw=raw_complete[raw_complete[year_col]==anno]
        raw=raw.sort_values(col_day)




        logging_textbox = st.empty()
  
        for day_iter in range(day,nn):
            with logging_textbox.container():
                st.write("Valuto la giornata {} dell'anno {}".format(day_iter,anno))
            df_period=raw[raw[col_day]<=day_iter] #il dataframe contiene il periodo da giornata 0 a adesso
            squadre_day=list(df_period.groupby(['SQUADRA']).mean().index)
            [avgnowch,avgdxdnowch]=champions_metrics(df_period,col_day=col_day)
            [ndnows,qtymaxnows]=team_metrics(df_period,squadre)
            [ndnows_sf,qtymaxnows_sf]=team_metrics(df_period,squadre,col_squadre='SFIDANTE')

            for squadra in squadre_day:
                line_df=df_period[df_period[col_day]==day_iter]
                line_team=line_df[line_df['SQUADRA']==squadra]
              
                #Media di pari negli ultimi 3 anni per campionato
                line_team[input[0]]=avg3yrch

                #Media di pari negli ultimi giorni da D=0 a D=now
                line_team[input[1]]=avgnowch
                try:
                    #Media di non pari negli ultimi 3 anni per squadra
                    line_team[input[2]]=nd3yrs.loc[squadra].values[0]
                except Exception as e:
                    st.write(":red[Sto avendo problemi con la squadra {}. Forse non c'era nel training, oppure l'hai scritto male!]".format(squadra))

                #Media di non pari negli ultimi giorni da D=0 a D=now
                line_team[input[3]]=ndnows[squadra]

                #Media di pari per giornata negli ultimi 3 anni per campionato
                line_team[input[4]]=avgdxd3yrch

                #Media di pari per giornata negli ultimi giorni da D=0 a D=now
                line_team[input[5]]=avgdxdnowch

                #Quantity di non pari longer negli ultimi 3 anni
                line_team[input[6]]=qtymax3yrs.loc[squadra].values[0]

                #Quantity di non pari da D=0 a D=now
                line_team[input[7]]=qtymaxnows.loc[squadra].values[0]
                
                #media e dev std di goal per la squadra
                line_team[input[8]]=df_period[df_period["SQUADRA"]==squadra]["N_GOAL"].mean()
                line_team[input[9]]=df_period[df_period["SQUADRA"]==squadra]["N_GOAL"].std()
                line_team[input[10]]=df_period[df_period["SQUADRA"]==squadra]["N_GOAL_SF"].mean()
                line_team[input[11]]=df_period[df_period["SQUADRA"]==squadra]["N_GOAL_SF"].std()

                #'AVG_ND_3Y_S_SFID':'Media di non-pareggi per squadra, calcolata sulle stagioni precedenti',\
                try:
                    line_team[input[12]]=nd3yrs_sf.loc[squadra].values[0]
                except Exception as e:
                    st.write(":red[Sto avendo problemi con la squadra {}. Forse non c'era nel training, oppure l'hai scritto male!]".format(squadra))

                #'AVG_ND_N_S_SFID':'Media di non-pareggi per squadra, calcolata sulla stagione attuale',\
                line_team[input[13]]=ndnows_sf[squadra]

                #'QTY_ND_3Y_S_SFID':'Media del periodo massimo per stagione di giornate consecutive senza pareggi per squadra, calcolata sulle stagioni precedenti',\
                line_team[input[14]]=qtymax3yrs_sf.loc[squadra].values[0]                
                
                #'QTY_ND_N_S_SFID':'Quantità di giornate consecutive senza pareggi per squadra sulla stagione attuale',\
                line_team[input[15]]=qtymaxnows_sf.loc[squadra].values[0]
                    



                int_df=pd.concat([int_df,line_team])

            int_df=int_df.dropna()
            final_df=int_df[int_df[col_day]==day] #final df contiene la sola riga del giorno x
            final_df=final_df.fillna(0)
            #per training il df è int_df
        if what=='pred':
            st.write('Ecco il dataset su cui faccio previsioni. Ho cancellato i valori nulli. Selezionerò solo le colonne che hai usato nel training.')
            download_excel(final_df,'Pre-training_dataset_Day{}'.format(day))
        elif what=='val':
            st.write("Ecco il dataset su cui faccio la validazione. Ho cancellato i valori nulli. Selezionerò solo le colonne che hai usato nel training.")
            download_excel(int_df,'Pre-training_dataset_whole')
        elif what=='train':


            st.write("Ecco il dataset su cui faccio l'allenamento. Ho riempito i valori nulli con 0. Selezionerò solo le colonne che hai scelto.")
            download_excel(int_df,'Pre-training_dataset_whole')
        else:
            st.write('Task non chiaro.')
 
            #in che giorno nella riga



        return(raw,final_df,int_df)
    
def prediction(save,output_choice,final_df,input_lower='na'):
        if input_lower=='na':
            [a,input_lower]=starting()
        st.write("\n:robot_face: E mo' predico. :robot_face:")

        ##Modello: predizioni per output
        #nome_modello= os.path.join(os.getcwd(), os.path.normpath('Modello_{}'.format(output_choice)))
 
        alg=save['Algorithm']
        input=save['Input']
        output_choice=save['Output']

        output_select=save['Predict_future']
        
        #st.write(dict)
        #alg=dict['Algorithm']
        #alg=uploaded_model['Algorithm']

        final_df['{}_pred'.format(output_choice)]=alg.predict(final_df[input])
        final_df['{}_prob0'.format(output_choice)]=alg.predict_proba(final_df[input])[:,0]
        final_df['{}_prob1'.format(output_choice)]=alg.predict_proba(final_df[input])[:,1]
        final_df=final_df.dropna()
        alg_w=alg
         ##Modello: predizioni per output, con meno input
        '''
        nome_modello= os.path.join(os.getcwd(), os.path.normpath('Modello_{}_lower_input'.format(output_choice)))
        dict=pickle.load(open(nome_modello, 'rb'))
        alg=dict['Algorithm']
        final_df['{}_lp_pred'.format(output_choice)]=alg.predict(final_df[input_lower])
        final_df['{}_lp_probA'.format(output_choice)]=alg.predict_proba(final_df[input_lower])[:,0]
        final_df['{}_lp_probB'.format(output_choice)]=alg.predict_proba(final_df[input_lower])[:,1]
        final_df=final_df.dropna()
        alg_lp=alg
        '''
        return(final_df,alg_w)
    
def talk(day_iter,output_choice,final_df):
        st.title('Risultati per la giornata {}'.format(day_iter))
        st.subheader("""Prima versione""")
        st.write('Questi risultati sono ottenuti con modelli allenati su questi input:')

        #ordino dalla probabilità di 0 più piccola alla più grande
        final_df=final_df.sort_values('{}_prob0'.format(output_choice))
        
        st.write('Valutando {}, nella giornata {} dovresti investire su: :moneybag:'.format(output_choice,day_iter))
        for ii in range(0,7):
            sq=final_df['SQUADRA'].iloc[ii]
            pred=final_df['{}_pred'.format(output_choice)].iloc[ii]
            prob=final_df['{}_prob1'.format(output_choice)].iloc[ii]
            oth=final_df['{}_prob0'.format(output_choice)].iloc[ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))

        st.write("___________________________________________")
        st.write('Valutando {}, le squadre con meno probabilità di pareggiare nella giornata {} sono: :sloth:'.format(output_choice,day_iter))
        for ii in range(1,8):
            sq=final_df['SQUADRA'].iloc[-ii]
            pred=final_df['{}_pred'.format(output_choice)].iloc[-ii]
            prob=final_df['{}_prob1'.format(output_choice)].iloc[-ii]
            oth=final_df['{}_prob0'.format(output_choice)].iloc[-ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))
        st.write('___________________________________________')
        if final_df['{}_prob1'.format(output_choice)].mean()==1:
            st.write('Mah, ste probabilità so tutte uguali a 1. Grazie al c:sparkles:...')



        
        '''
        st.subheader("""Seconda versione""")
        st.write('Questi risultati sono ottenuti con modelli allenati su questi input:')
        st.write(input_lower)
        final_df=final_df.sort_values('{}_lp_probA'.format(output_choice))
        
        st.write('Valutando {}, nella giornata {} dovresti investire su: :moneybag:'.format(output_choice,day_iter))
        for ii in range(0,7):
            sq=final_df['SQUADRA'].iloc[ii]
            pred=final_df['{}_lp_pred'.format(output_choice)].iloc[ii]
            prob=final_df['{}_lp_probB'.format(output_choice)].iloc[ii]
            oth=final_df['{}_lp_probA'.format(output_choice)].iloc[ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))

        st.write("___________________________________________")
        st.write('Valutando {}, le squadre con meno probabilità di pareggiare nella giornata {} sono: :sloth:'.format(output_choice,day_iter))
        for ii in range(0,7):
            sq=final_df['SQUADRA'].iloc[-ii]
            pred=final_df['{}_lp_pred'.format(output_choice)].iloc[-ii]
            prob=final_df['{}_lp_probB'.format(output_choice)].iloc[-ii]
            oth=final_df['{}_lp_probA'.format(output_choice)].iloc[-ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))
        st.write('___________________________________________')
        if final_df['{}_lp_probB'.format(output_choice)].mean()==1:
            st.write('Mah, ste probabilità so di nuovo tutte uguali a 1. Grazie al c:sparkles:...')
        else:
            st.write('Qui almeno abbiamo probabilità diverse')
        '''



