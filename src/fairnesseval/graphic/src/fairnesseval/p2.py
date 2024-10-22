import streamlit as st
import pandas as pd
import numpy as np
import logging
from fairnesseval import utils_experiment_parameters as exp_params
from fairnesseval.run import launch_experiment_by_config

# List of datasets and models
all_datasets = exp_params.ACS_dataset_names + ['adult', 'compas', 'german']
model_list = [
    'LogisticRegression',
    'expgrad',
    'ThresholdOptimizer',
    'ZafarDI',
    'ZafarEO',
    'Feld'
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Function to safely evaluate text input for parameters and fractions
def eval_none(x):
    try:
        return eval(x)
    except Exception as e:
        return None

# Streamlit App
st.title("Experiment Definition and Execution")

# Experiment ID
experiment_id = st.text_input('Experiment Code', 'demo.default.test')

# Dataset Selection
st.markdown("**Dataset Selection**")
selected_datasets = st.multiselect('Choose datasets', all_datasets, default=[all_datasets[-3]])

# Model Selection
st.markdown("**Model Selection**")
selected_models = st.multiselect('Choose models', model_list, default=[model_list[0]])

# Model Parameters
st.markdown("**Model Parameters**")
model_parameters = st.text_area('Enter parameters as key-value pairs (e.g., {"param1": value1, "param2": value2})', '')

# Train Fractions
train_fractions = st.text_area('Train fractions (as list pairs e.g., [0.016, 0.063, 0.251, 1.])', '')

# Random Seed
random_seed = st.number_input('Random seed', value=42)

# Button to trigger experiment
if st.button('Run experiment'):
    # Prepare experiment configuration
    experiment_conf = {
        'experiment_id': experiment_id,
        'dataset_names': selected_datasets,
        'model_names': selected_models,
        'random_seeds': [random_seed],
        'model_params': eval_none(model_parameters),
        'train_fractions': eval_none(train_fractions),
        'results_path': './demo_results',
        'params': ['--debug']  # Placeholder for additional parameters
    }

    # Remove empty values from config
    experiment_conf = {k: v for k, v in experiment_conf.items() if v}

    # Display the config
    st.write("Experiment configuration:", experiment_conf)

    # Try running the experiment
    try:
        launch_experiment_by_config(experiment_conf)
        st.success("Experiment executed successfully!")
    except Exception as e:
        st.error(f"Error during experiment execution: {str(e)}")
