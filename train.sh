
export CUDA_VISIABLE_DEVICES=0,1
python train.py --dataroot ./datasets/hazy --name priorcyclegan --model cycle_gan --lambda_identity 0 \
--no_flip