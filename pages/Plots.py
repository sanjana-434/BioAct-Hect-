import streamlit as st
import os

def plots():
    pltlist = os.listdir('Plots')
    for i in pltlist:
        st.image("Plots\\" + i)

plots()