FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


COPY ./requirements.txt requirements.txt
COPY ./install.sh install.sh
RUN ./install.sh

WORKDIR .

