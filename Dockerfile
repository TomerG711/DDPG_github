FROM nvcr.io/nvidia/pytorch:21.04-py3
COPY . /opt/DDPG
WORKDIR /opt/DDPG
RUN pip install lpips

CMD python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0.05 \
-i DDPG_celeba_sr_0.05  --inject_noise 1 --gamma 10 --zeta 0.8 --eta_tilde 0.3  --step_size_mode 2 --seed 1234  \
--operator_imp SVD --deg_scale 4


#--config imagenet_256.yml --path_y imagenet
#--config celeba_hq.yml --path_y celeba_hq