FROM quay.io/fenicsproject/dev:latest

# Instalar pacotes adicionais necessários
RUN pip3 install notebook matplotlib numpy

# Copiar seu código para o diretório de trabalho
COPY . /home/fenics/shared

WORKDIR /home/fenics/shared

# Comando para iniciar o Jupyter Notebook
CMD ["jupyter-notebook", "--ip", "0.0.0.0", "--no-browser"]
