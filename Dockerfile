FROM nvcr.io/nvidia/pytorch:21.04-py3
COPY . /opt/DDPG
WORKDIR /opt/DDPG
RUN pip install lpips


#CMD ./eval.sh

CMD python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0 \
-i DDPG_celeba_sr_bicubic_sigma_y_0 --inject_noise 1 --zeta 0.7 --step_size_mode 0 \
--deg_scale 4 --operator_imp SVD


#--config imagenet_256.yml --path_y imagenet
#--config celeba_hq.yml --path_y celeba_hq