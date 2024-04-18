import streamlit as st
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import os





def make_predictions():
    pass


# Set title and description
st.title("BioAct-Het and Canonical GCN Prediction")
import streamlit as st
import pandas as pd

# Define the information for the Sider dataset
sider_info = {
    'Dataset': ['Sider'],
    'Bioactivity Type': ['Side Effects'],
    'Compounds': [1427],
    'Bioactivity Classes': [27],
    'Positive Samples': [21868],
    'Negative Samples': [16661]
}

# Define the information for the Tox21 dataset
tox21_info = {
    'Dataset': ['Tox21'],
    'Bioactivity Type': ['Toxicity'],
    'Compounds': [7831],
    'Bioactivity Classes': [12],
    'Positive Samples': [5862],
    'Negative Samples': [72084]
}

# Convert the dictionaries to DataFrames
df_sider_info = pd.DataFrame(sider_info)
df_tox21_info = pd.DataFrame(tox21_info)

# Display the information for the Sider dataset
st.header("Sider Dataset Information")
st.dataframe(df_sider_info)

# Display the information for the Tox21 dataset
st.header("Tox21 Dataset Information")
st.dataframe(df_tox21_info)

#plots()
#displayAccuracy()

