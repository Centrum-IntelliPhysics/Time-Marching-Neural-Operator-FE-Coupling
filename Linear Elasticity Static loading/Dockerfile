FROM dolfinx/dolfinx:v0.9.0
WORKDIR /root

#Installs necessary dependencies
COPY ./requirements.txt ./requirements.txt
RUN ["python", "-m", "pip", "install", "-r", "requirements.txt"]