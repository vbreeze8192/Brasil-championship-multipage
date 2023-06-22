# Contents of ~/my_app/main_page.py
import numpy as np
import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")
st.write('Scegli cosa vuoi vedere.')

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
    txt=int(np.round(np.random(0, len(txt_choice))))
    st.write(txt)
