# Cycle Consistent Adversarial Domain Adaptation (CyCADA)
A [pytorch](http://pytorch.org/) implementation of [CyCADA](https://arxiv.org/pdf/1711.03213.pdf). 

If you use this code in your research please consider citing

>@inproceedings{Hoffman_cycada2017,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; authors = {Judy Hoffman and Eric Tzeng and Taesung Park and Jun-Yan Zhu,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and Phillip Isola and Kate Saenko and Alexei A. Efros and Trevor Darrell},<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          title = {CyCADA: Cycle Consistent Adversarial Domain Adaptation},<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          booktitle = {International Conference on Machine Learning (ICML)},<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          year = 2018<br>
}

## Setup
* Check out the repo (recursively will also checkout the CyCADA fork of the CycleGAN repo).<br>
`git clone --recursive https://github.com/jhoffman/cycada_release.git cycada`
* Install python requirements
    * pip install -r requirements.txt
    
## Train image adaptation only (digits)
* Image adaptation builds on the work on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The submodule in this repo is a fork which also includes the semantic consistency loss. 
* Pre-trained image results for digits may be downloaded here
  * [SVHN as MNIST](https://people.eecs.berkeley.edu/~jhoffman/cycada/svhn2mnist.zip) (114MB)
  * [MNIST as USPS](https://people.eecs.berkeley.edu/~jhoffman/cycada/mnist2usps.zip) (6MB)
  * [USPS as MNIST](https://people.eecs.berkeley.edu/~jhoffman/cycada/usps2mnist.zip) (3MB)
* Producing SVHN as MNIST 
   * For an example of how to train image adaptation on SVHN->MNIST, see `cyclegan/train_cycada.sh`. From inside the `cyclegan` subfolder run `train_cycada.sh`. 
   * The snapshots will be stored in `cyclegan/cycada_svhn2mnist_noIdentity`. Inside `test_cycada.sh` set the epoch value to the epoch you wish to use and then run the script to generate 50 transformed images (to preview quickly) or run `test_cycada.sh all` to generate the full ~73K SVHN images as MNIST digits. 
   * Results are stored inside `cyclegan/results/cycada_svhn2mnist_noIdentity/train_75/images`. 
   * Note we use a dataset of mnist_svhn and for this experiment run in the reverse direction (BtoA), so the source (SVHN) images translated to look like MNIST digits will be stored as `[label]_[imageId]_fake_B.png`. Hence when images from this directory will be loaded later we will only images which match that naming convention.

## Train feature adaptation only (digits)
* The main script for feature adaptation can be found inside `scripts/train_adda.py`
* Modify the data directory you which stores all digit datasets (or where they will be downloaded)

## Train feature adaptation following image adaptation
* Use the feature space adapt code with the data and models from image adaptation
* For example: to train for the SVHN to MNIST shift, set `src = 'svhn2mnist'` and `tgt = 'mnist'` inside `scripts/train_adda.py` 
* Either download the relevant images above or run image space adaptation code and extract transferred images

## Train Feature Adaptation for Semantic Segmentation
* Download [GTA as CityScapes](http://efrosgans.eecs.berkeley.edu/cyclegta/cyclegta.zip) images (16GB).
* Download [GTA DRN-26 model](https://people.eecs.berkeley.edu/~jhoffman/cycada/drn26-gta5-iter115000.pth)
* Download [GTA as CityScapes DRN-26 model](https://people.eecs.berkeley.edu/~jhoffman/cycada/drn26-cyclegta5-iter115000.pth)
* Adapt using `scripts/train_fcn_adda.sh`
   * Choose the desired `src` and `tgt` and `datadir`. Make sure to download the corresponding base model and data. 
   * The final DRN-26 CyCADA model from GTA to CityScapes can be downloaded [here](https://people.eecs.berkeley.edu/~jhoffman/cycada/drn26_cycada_cyclegta2cityscapes.pth)

