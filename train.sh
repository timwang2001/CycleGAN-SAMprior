export CUDA_VISIABLE_DEVICES=0,1
python train.py --dataroot ./datasets/hazy --name maps_cyclegan --model cycle_gan --lambda_identity 0 \
--num_threads 1