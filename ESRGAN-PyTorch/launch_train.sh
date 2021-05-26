python -u train.py dataset_srgan2 \
        -j 8 \
        --start-psnr-epoch 55 --psnr-epochs 55\
        -b 32 \
        --resume_psnr /home/ubuntu/ESRGAN-PyTorch/weights/PSNR_epoch55.pth \
        --gpu 0
