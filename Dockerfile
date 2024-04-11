FROM nvcr.io/nvidia/pytorch:21.04-py3

COPY . /opt/DDPG

WORKDIR /opt/DDPG

RUN pip install lpips

#CMD ./evaluation_DDPG.sh
CMD ./evaluation_IDPG.sh