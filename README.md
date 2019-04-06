# superresolution_gan

Chainer v5.1 implementation of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)  
[Original source code](https://github.com/Hi-king/superresolution_gan) was implemented by Hi-king  
This source code was updated to work with Chainer v5.1

### Training

```
python train.py \
  --gpu=0  \
  --pretrained_generator generator_model_3008000.npz \
  --src ./src
  --batchsize=16 \
  --k_mse=0.0001 \
  --k_adversarial=0.00001 \
  --outdirname output
  --vgg
```
