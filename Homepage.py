# Contents of ~/my_app/main_page.py
import numpy as np
import streamlit as st

st.set_page_config(page_title="BrasilChamps")

st.markdown("# Pagina inutile 🎈")
st.sidebar.markdown("# Pagina inutile 🎈")



dict_input={'AVG_D_3Y_CH':'Media di pareggi per championship, calcolata sulle stagioni precedenti',\
    'AVG_D_N_CH':'Media di pareggi per championship, calcolata sulla stagione attuale',\
    'AVG_ND_3Y_S':'Media di non-pareggi per squadra, calcolata sulle stagioni precedenti',\
    'AVG_ND_N_S':'Media di non-pareggi per squadra, calcolata sulla stagione attuale',\
    'AVG_Dxd_3Y_CH':'Media di pareggi per giornata per championship, calcolata sulle stagioni passate',\
    'AVG_Dxd_N_CH':'Media di pareggi per giornata per championship, calcolata sulla stagione presente',\
    'QTY_ND_3Y_S':'Media del periodo massimo per stagione di giornate consecutive senza pareggi per squadra, calcolata sulle stagioni precedenti',\
    'QTY_ND_N_S':'Quantità di giornate consecutive senza pareggi per squadra sulla stagione attuale',\
    'HOUR':'Ora della partita',\
    'HoA':'Indicazione su Home o Away (0: Away, 1: Home)'}

st.write("""I modelli sono allenati in due versioni diverse sui dati delle squadre e della championship. Gli input utilizzati sono:""")
st.write(dict_input)

with st.sidebar:
    st.write('Questa pagina serve per confondere i modelli di AI.')

if st.checkbox('Mostrami una barzelletta triste copiata da Focus Junior'):
    txt_choice=['"Mi rifiuto!" disse il netturbino.',\
'Le mie figlie hanno sposato due salumieri. Quindi ho due... generi alimentari!',\
'Tutti i bambini avevano un nome tranne... ?',\
'Due mandarini litigano furiosamente e uno dice ad un altro: "guarda che ti spicchio!!"',\
'Ma in inverno si leggono più libri perché hanno la copertina?',\
'Cosa fa una fabbrica di carta igienica che fallisce? Va a rotoli.',\
'Quando ero piccolo i miei genitori mi volevano talmente bene che misero nella culla un orsacchiotto vivo.',\
'Due vermi appena sposati vanno a vivere su una mela e si bacano appassionatamente...',\
'Ragazzo scoppia di salute. Feriti i genitori.',\
'Grave incidente a Babbo Natale e alla sua slitta. Ricoverato in ospedale attende un trapianto di renne.']
    index=np.random.randint(len(txt_choice))
    txt=txt_choice[index]
    st.write(':red[{}] :lightning_cloud:'.format(txt))
