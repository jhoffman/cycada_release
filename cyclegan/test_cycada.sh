if [ "$1" == "all" ]; then
    how_many=100000
else
    how_many=50
fi

model=cycada_svhn2mnist_noIdentity
epoch=75
CUDA_VISIBLE_DEVICES=0 python test.py --name ${model} \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic \
    --no_flip --batchSize 100 \
    --dataset_mode mnist_svhn --dataroot /x/jhoffman/ \
    --which_direction BtoA \
    --phase train \
    --how_many ${how_many} \
    --which_epoch ${epoch}

