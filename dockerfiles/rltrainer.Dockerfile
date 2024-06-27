FROM bitnami/pytorch

#USER root
# update bins
# RUN apt-get update && apt-get -y install build-essential curl git \
# libexpat1-dev libfreetype6-dev

WORKDIR /app

COPY ./poke-env/requirements.txt .

RUN pip3 install -r requirements.txt