FROM jupyter/scipy-notebook:latest

# Install FEniCS and other dependencies
RUN conda install -c conda-forge fenics

# Copy your code to the working directory
COPY . /home/jovyan/

WORKDIR /home/jovyan/

# The base image already has the correct CMD to start the notebook
