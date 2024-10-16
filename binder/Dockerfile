# FROM quay.io/fenicsproject/stable:current
FROM quay.io/fenicsproject/dev

# Instalar pacotes adicionais necessários
RUN pip3 install matplotlib numpy

# Copiar seu código para o diretório de trabalho
COPY . /home/fenics/shared

WORKDIR /home/fenics/shared
