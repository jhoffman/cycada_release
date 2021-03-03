CUDA_VISIBLE_DEVICES=0 python test.py --name gta2cityscapes \
    --resize_or_crop="scale_width_and_crop" \
    --loadSize=1024 --fineSize=1024 \
    --model cycle_gan \
    --batchSize 1 \
    --dataset_mode unaligned --dataroot /mnt/data/cyclegan_data/ \
    --which_direction AtoB
