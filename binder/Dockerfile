FROM quay.io/fenicsproject/stable:current

# Instalar pacotes adicionais
RUN pip3 install notebook matplotlib numpy

# Copiar seu código para o diretório de trabalho
COPY . /home/fenics/shared

WORKDIR /home/fenics/shared
