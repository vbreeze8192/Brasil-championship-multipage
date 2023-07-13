import streamlit as st

st.set_page_config(page_title="Un cacchio")

st.markdown("# Pagina inutile n2 🎈")
st.sidebar.markdown("# Pagina inutile n2 🎈")

with st.sidebar:
    st.write('Questa pagina serve per confondere gli sviluppatori.')


st.write("# This works:")

if "Predici" not in st.session_state:
    st.session_state["Predici"] = False

if "button2" not in st.session_state:
    st.session_state["button2"] = False

if "button3" not in st.session_state:
    st.session_state["button3"] = False

if st.button("Predici"):
    st.session_state["Predici"] = not st.session_state["Predici"]
    st.write('Clicking button 1')

if st.session_state["Predici"]:
    if st.button("Button2"):
        st.session_state["button2"] = not st.session_state["button2"]

if st.session_state["Predici"] and st.session_state["button2"]:
    if st.button("Button3"):
        # toggle button3 session state
        st.session_state["button3"] = not st.session_state["button3"]

if st.session_state["button3"]:
    st.write("**Button3!!!**")


# Print the session state to make it easier to see what's happening
st.write(
    f"""
    ## Session state:
    {st.session_state["Predici"]=}

    {st.session_state["button2"]=}

    {st.session_state["button3"]=}
    """
)

