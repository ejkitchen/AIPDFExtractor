# Base image from NVIDIA's Triton Server with Python 3 support
FROM nvcr.io/nvidia/tritonserver:24.04-py3

# Avoid prompts during the SSH connection
ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

# Install required system dependencies, Python 3.10, and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-distutils \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* get-pip.py

# Set Python 3.10 as the default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set temporary directories to prevent model caching in the image
ENV HF_HOME=/tmp/ \
    TORCH_HOME=/tmp/

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies without caching to reduce image size
RUN pip3 install --no-cache-dir -r requirements.txt

# Download pretrained models in a single step to reduce layers
RUN python -c 'from deepsearch_glm.utils.load_pretrained_models import load_pretrained_nlp_models; load_pretrained_nlp_models(verbose=True);'
RUN python -c 'from docling.document_converter import DocumentConverter; artifacts_path = DocumentConverter.download_models_hf(force=True);'

RUN rm -rf /root/.cache /tmp/* 

# Set the number of threads to avoid congestion in container environments
ENV OMP_NUM_THREADS=32

# Copy minimal.py to the root directory of the container
WORKDIR /
COPY minimal.py .

# Run the minimal.py script from the root directory so it downloads all the models and files it needs into the container
RUN python minimal.py

# Return to the app directory
WORKDIR /app

# The CMD instruction is not needed here as it's specified in docker-compose.yml