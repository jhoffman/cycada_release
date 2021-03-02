CUDA_VISIBLE_DEVICES=0 python ./cyclegan/train.py --name gta2cityscapes \
    --resize_or_crop="scale_width_and_crop" \
    --loadSize=1024 --fineSize=400 \
    --model cycle_gan \
    --lambda_identity 1.0 \
    --batchSize 8 \
    --dataset_mode unaligned --dataroot /mnt/data/cyclegan_data/ \
    --which_direction AtoB
