import streamlit as st
import logging
import time
import io
import time
import os
from fairnesseval import utils_experiment_parameters as exp_params
from fairnesseval.run import launch_experiment_by_config
from datetime import datetime

# Create a handler to write logs to a buffer
class StreamlitLogger(io.StringIO):
    def __init__(self, log_area):
        super().__init__()
        self.log_area = log_area  # Area where logs will be displayed
        self.log_data = ""  # String to accumulate log messages

    def write(self, message):
        # Filter out any empty messages
        if message.strip():
            # Accumulate logs and update log area
            self.log_data += message + "\n"
            self.log_area.markdown(f"```\n{self.log_data}\n```")  # Update log display

# Configure the logger to use the Streamlit buffer
def setup_logging(log_area):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Streamlit handler
    streamlit_handler = logging.StreamHandler(StreamlitLogger(log_area))
    streamlit_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    streamlit_handler.setFormatter(formatter)

    # Clear previous handlers and add the new one
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(streamlit_handler)

    return logger


def pagina1():
    # List of datasets and models
    all_datasets = exp_params.ACS_dataset_names + ['adult', 'compas', 'german']
    model_list = [
        'LogisticRegression',
        'Expgrad',
        'ThresholdOptimizer',
        'ZafarDI',
        'ZafarEO',
        'Feld'
    ]

    def eval_none(x):
        if not x.strip():
            return None
        try:
            return eval(x)
        except Exception as e:
            logger.error(f"Error evaluating input {x}: {str(e)}")
            return None

    # Streamlit title
    st.title("Experiment definition and execution")

    # Dataset selection
    selected_datasets = st.multiselect(
        'Datasets',
        sorted(all_datasets),
        default=[sorted(all_datasets)[-3]]
    )

    # Model selection
    selected_models = st.multiselect(
        'Models',
        sorted(model_list),
        default=[sorted(model_list)[2]]
    )

    # Model parameters input
    model_parameters = st.text_area('Model parameters: enter parameters as key-value pairs (e.g., {"param1": value1, "param2": value2}). These values will be used as parameters of the model. (Optional)', '')

    # Train fractions input
    train_fractions = st.text_input('Train fractions: enter training fractions (e.g., [0.016, 0.063, 0.251, 1.]) (Optional)', '')

    # Random seed input
    random_seed = st.text_input('Random Seed: enter a value or a list of values (e.g. [41,42,23]) (Required)', '')

    # Experiment ID
    experimentID = st.text_input('Experiment ID: enter the name of the experiment (Required)', '')

    # Button to run the experiment
    if st.button('Run Experiment'):
        # Check if experiment ID exists in the demo_results folder
        results_path = './demo_results'
        experiment_path = os.path.join(results_path, experimentID)

        if os.path.exists(experiment_path):
            st.warning(f"Experiment ID '{experimentID}' already exists. Please choose a different name.")
        else:
            # Create an empty area to display logs below the button
            log_area = st.empty()

            # Configure loggers for Streamlit
            logger = setup_logging(log_area)

            # Experiment configuration
            experiment_conf = {
                'experiment_id': experimentID,
                'dataset_names': selected_datasets,
                'model_names': selected_models,
                'random_seed': eval_none(random_seed),
                'model_params': eval_none(model_parameters),
                'train_fractions': eval_none(train_fractions),
                'results_path': results_path,
                'params': ['--debug']  # Placeholder for other parameters
            }

            # Log the configuration before cleaning it up (formatted with new lines)
            logger.info("Experiment configuration before cleanup (raw):")
            for key, value in experiment_conf.items():
                logger.info(f"{key}: {value}")

            # Remove empty values from the configuration
            experiment_conf = {k: v for k, v in experiment_conf.items() if v}

            # Log the final cleaned-up configuration (formatted with new lines)
            logger.info("Final experiment configuration (after cleanup):")
            for key, value in experiment_conf.items():
                logger.info(f"{key}: {value}")

            # Attempt to run the experiment
            try:
                logger.info("Starting experiment...")

                # Simulate logging during the experiment
                for i in range(5):
                    time.sleep(1)  # Simulate the execution time of the experiment
                    logger.info(f"Running... Step {i + 1}/5")

                # Execute the experiment
                launch_experiment_by_config(experiment_conf)

                logger.info("Experiment successfully completed!")
            except Exception as e:
                logger.error(f"Error during experiment execution: {str(e)}")

# Run the function pagina1
#pagina1()
