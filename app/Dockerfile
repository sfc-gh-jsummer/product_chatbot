FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

# Set the working directory
WORKDIR /workspace/

RUN mkdir /workspace/.local /workspace/.cache && chmod 777 -R /workspace
COPY --chmod=777 ./ ./ 

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Run Jupyter Notebook on container startup
ENTRYPOINT ["./entrypoint.sh"]
 



