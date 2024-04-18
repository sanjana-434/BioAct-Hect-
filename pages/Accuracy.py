import streamlit as st
import pandas as pd

def displayAccuracy():
    # Load your dataset and preprocess it
    # Return necessary data for making predictions
    # Define the results for sider dataset with Association_based method and BioAct-Het and AttentiveFp GCN model
    sider_results_attentivefp_association = {
        'Dataset': ['Sider'],
        'Accuracy': [0.8287004888057709],
        'AUC': [0.9111991882324219],
        'Method': ['Association_based'],
        'Model': ['BioAct-Het and AttentiveFp GCN']
    }

    # Define the results for sider dataset with Association_based method and BioAct-Het and Canonical GCN model
    sider_results_canonical_association = {
        'Dataset': ['Sider'],
        'Accuracy': [0.8200058460235595],
        'AUC': [0.9034979045391083],
        'Method': ['Association_based'],
        'Model': ['BioAct-Het and Canonical GCN']
    }

    # Define the results for sider dataset with BioActivity_based method and BioAct-Het and AttentiveFp GCN model
    sider_results_attentivefp_bioactivity = {
        'Dataset': ['Sider'],
        'Accuracy': [0.7428], # Placeholder values
        'AUC': [0.8160], # Placeholder values
        'Method': ['BioActivity_based'],
        'Model': ['BioAct-Het and AttentiveFp GCN']
    }

    # Define the results for sider dataset with BioActivity_based method and BioAct-Het and Canonical GCN model
    sider_results_canonical_bioactivity = {
        'Dataset': ['Sider'],
        'Accuracy': [0.7540], # Placeholder values
        'AUC': [0.8060], # Placeholder values
        'Method': ['BioActivity_based'],
        'Model': ['BioAct-Het and Canonical GCN']
    }

    # Define the results for sider dataset with Drug_based method and BioAct-Het and AttentiveFp GCN model
    sider_results_attentivefp_drug = {
        'Dataset': ['Sider'],
        'Accuracy': [0.7726], # Placeholder values
        'AUC': [0.848], # Placeholder values
        'Method': ['Drug_based'],
        'Model': ['BioAct-Het and AttentiveFp GCN']
    }

    # Define the results for sider dataset with Drug_based method and BioAct-Het and Canonical GCN model
    sider_results_canonical_drug = {
        'Dataset': ['Sider'],
        'Accuracy': [0.7619], # Placeholder values
        'AUC': [0.8192], # Placeholder values
        'Method': ['Drug_based'],
        'Model': ['BioAct-Het and Canonical GCN']
    }

    # Define the results for Tox21 dataset with Association_based method and BioAct-Het and AttentiveFp GCN model
    tox21_results_attentivefp_association = {
        'Dataset': ['Tox21'],
        'Accuracy': [0.8927],
        'AUC': [0.8909],
        'Method': ['Association_based'],
        'Model': ['BioAct-Het and AttentiveFp GCN']
    }

    # Define the results for Tox21 dataset with Association_based method and BioAct-Het and Canonical GCN model
    tox21_results_canonical_association = {
        'Dataset': ['Tox21'],
        'Accuracy': [0.8615],
        'AUC': [0.8577],
        'Method': ['Association_based'],
        'Model': ['BioAct-Het and Canonical GCN']
    }

    # Define the results for Tox21 dataset with BioActivity_based method and BioAct-Het and AttentiveFp GCN model
    tox21_results_attentivefp_bioactivity = {
        'Dataset': ['Tox21'],
        'Accuracy': [0.9401], # Placeholder values
        'AUC': [0.8498], # Placeholder values
        'Method': ['BioActivity_based'],
        'Model': ['BioAct-Het and AttentiveFp GCN']
    }

    # Define the results for Tox21 dataset with BioActivity_based method and BioAct-Het and Canonical GCN model
    tox21_results_canonical_bioactivity = {
        'Dataset': ['Tox21'],
        'Accuracy': [0.9377], # Placeholder values
        'AUC': [0.8412], # Placeholder values
        'Method': ['BioActivity_based'],
        'Model': ['BioAct-Het and Canonical GCN']
    }

    # Define the results for Tox21 dataset with Drug_based method and BioAct-Het and AttentiveFp GCN model
    tox21_results_attentivefp_drug = {
        'Dataset': ['Tox21'],
        'Accuracy': [0.8364], # Placeholder values
        'AUC': [0.7910], # Placeholder values
        'Method': ['Drug_based'],
        'Model': ['BioAct-Het and AttentiveFp GCN']
    }

    # Define the results for Tox21 dataset with Drug_based method and BioAct-Het and Canonical GCN model
    tox21_results_canonical_drug = {
        'Dataset': ['Tox21'],
        'Accuracy': [0.8655], # Placeholder values
        'AUC': [0.7887], # Placeholder values
        'Method': ['Drug_based'],
        'Model': ['BioAct-Het and Canonical GCN']
    }

    # Convert the dictionaries to DataFrames
    df_tox21_attentivefp_association = pd.DataFrame(tox21_results_attentivefp_association)
    df_tox21_canonical_association = pd.DataFrame(tox21_results_canonical_association)
    df_tox21_attentivefp_bioactivity = pd.DataFrame(tox21_results_attentivefp_bioactivity)
    df_tox21_canonical_bioactivity = pd.DataFrame(tox21_results_canonical_bioactivity)
    df_tox21_attentivefp_drug = pd.DataFrame(tox21_results_attentivefp_drug)
    df_tox21_canonical_drug = pd.DataFrame(tox21_results_canonical_drug)

    # Concatenate the DataFrames for Tox21 dataset
    df_tox21_combined = pd.concat([df_tox21_attentivefp_association, df_tox21_canonical_association,
                                df_tox21_attentivefp_bioactivity, df_tox21_canonical_bioactivity,
                                df_tox21_attentivefp_drug, df_tox21_canonical_drug], ignore_index=True)

   


    # Convert the dictionaries to DataFrames
    df_sider_association_attentivefp = pd.DataFrame(sider_results_attentivefp_association)
    df_sider_association_canonical = pd.DataFrame(sider_results_canonical_association)
    df_sider_bioactivity_attentivefp = pd.DataFrame(sider_results_attentivefp_bioactivity)
    df_sider_bioactivity_canonical = pd.DataFrame(sider_results_canonical_bioactivity)
    df_sider_drug_attentivefp = pd.DataFrame(sider_results_attentivefp_drug)
    df_sider_drug_canonical = pd.DataFrame(sider_results_canonical_drug)

    # Concatenate the DataFrames for sider dataset
    df_sider_combined = pd.concat([df_sider_association_attentivefp, df_sider_association_canonical,
                                df_sider_bioactivity_attentivefp, df_sider_bioactivity_canonical,
                                df_sider_drug_attentivefp, df_sider_drug_canonical], ignore_index=True)

    # Display the results for sider dataset
    st.subheader("Results for Sider dataset:")
    st.dataframe(df_sider_combined)
    # Display the results for Tox21 dataset
    st.subheader("Results for Tox21 dataset:")
    st.dataframe(df_tox21_combined)


displayAccuracy()