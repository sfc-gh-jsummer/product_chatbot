#!/bin/bash
# For Jupyter only - remove once validated
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &
# tail -f /dev/null # Keeps container running
streamlit run app.py