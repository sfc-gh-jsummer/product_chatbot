FROM jupyter/base-notebook:python-3.11

# Install the dependencies
RUN pip install requests weaviate-client==3.21.0 snowflake-connector-python pandas snowflake-snowpark-python[pandas]

# Set the working directory
WORKDIR /workspace/

# Expose Jupyter Notebook port
EXPOSE 8888

# Copy the working files to the container's working directory
RUN mkdir /workspace/.local /workspace/.cache && chmod 777 -R /workspace
COPY --chmod=777 vectorize.ipynb class_obj.json ./ 

# Run Jupyter on container startup
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]