# SPAINAI HACKATON 2020: COMPUTER VISION CHALLENGE

I'm using this repository to collect the main code used for the Computer Vision challenge from SpainAI's Hackaton 2020. 

## WHAT WAS THE CHALLENGE ABOUT

The challenge consisted on generating high resolution images (2400px) from low resolution images (600px). The images are mainly paintings, therefore the texture of the generated images is very important. The score for the competition is [SSIM](https://en.wikipedia.org/wiki/Structural_similarity) (Structural Similarity Index Measure).

This kind of problem of image generation is usually approached using [GANs Architectures](https://en.wikipedia.org/wiki/Generative_adversarial_network) (Generative Adversarial Networks). I used 2 different types of architectures, as I'll explain now.

## GENERAL CODE SUMMARY

The code you can find here is mainly for processing the data and training the Super Resolution GAN, not the Enhanced Super Resolution GAN, which works much better. For this last one, I'll later explain how to train it. 
With the scripts [create_data_folders.py](create_data_folders.py), [split_data_multiprocessing.py](split_data_multiprocessing.py), [split_images_in_patches.py](split_images_in_patches.py), and [create_validation.py](create_validation.py) we can create the data folders as needed for being able to train the GANs.

## MY SOLUTION

### 1. SRGAN

The [SRGAN](https://arxiv.org/abs/1609.04802) was developed in 2016. There's an implementation of it in Tensorflow [here](https://github.com/idealo/image-super-resolution). I've made a version of the package in [image-super-resolution/](image-super-resolution/). The results with this model are very poor and Tensorflow has much trouble with GPU memory management, therefore I don't recomment anyone using it. I tried with many different versions of Tensorflow etc, but couldn't make it fit a sufficiently large batch size. Aside from that, the algorithm doesn't learn as expected and its code is more difficult to modify. Using this, I obtained a **score of 0.31**.

![Alt text](imgs/SRGAN.png?raw=true "SRGAN")



### 2. ESRGAN

The [Enhanced Super Resolution Gan (ESRGAN)](https://arxiv.org/abs/1809.00219) is published in 2018 and presents some improvements to the previous SRGAN such as:
1. Architecture Improvement
2. Data Augmentation (Random flips)
3. Generator pre-training -> Better texture, accelerates convergence of the GAN.
4. No BatchNormalization.

You can use the script [ESRGAN-PyTorch/launch_train.sh](ESRGAN-PyTorch/launch_train.sh) for running training over the competition dataset. The script [ESRGAN-PyTorch/get_submission.py](ESRGAN-PyTorch/get_submission.py) is used for getting a submission once we have the model trained.

![Alt text](imgs/ESRGAN.png?raw=true "ESRGAN")


## EXAMPLES

In this section you will find some examples of the images generated, as well as their Low Resolution and real High Resolution Versions.

### EXAMPLE 1

* #### **LR** (LOW RESOLUTION)
![Alt text](imgs/lr/candidate_0133.png?raw=true "lr1")

* #### **SR** (SUPER-RESOLUTION)
![Alt text](imgs/sr/candidate_0133.png?raw=true "sr1")

* #### **HR** (HIGH RESOLUTION)
![Alt text](imgs/hr/candidate_0133.png?raw=true "hr1")


### EXAMPLE 2

* #### **LR** 
![Alt text](imgs/lr/candidate_0421.png?raw=true "lr2")

* #### **SR**
![Alt text](imgs/sr/candidate_0421.png?raw=true "sr2")

* #### **HR**
![Alt text](imgs/hr/candidate_0421.png?raw=true "hr2")


### EXAMPLE 3

* #### **LR**
![Alt text](imgs/lr/candidate_0847.png?raw=true "lr3")

* #### **SR**
![Alt text](imgs/sr/candidate_0847.png?raw=true "sr3")

* #### **HR**
![Alt text](imgs/hr/candidate_0847.png?raw=true "hr3")


### EXAMPLE 4

* #### **LR**
![Alt text](imgs/lr/candidate_0921.png?raw=true "lr4")

* #### **SR**
![Alt text](imgs/sr/candidate_0921.png?raw=true "sr4")

* #### **HR**
![Alt text](imgs/hr/candidate_0921.png?raw=true "hr4")


### EXAMPLE 5

* #### **LR**
![Alt text](imgs/lr/candidate_0927.png?raw=true "lr5")

* #### **SR**
![Alt text](imgs/sr/candidate_0927.png?raw=true "sr5")

* #### **HR**
![Alt text](imgs/hr/candidate_0927.png?raw=true "hr5")


### EXAMPLE 6

* #### **LR**
![Alt text](imgs/lr/candidate_1052.png?raw=true "lr6")

* #### **SR**
![Alt text](imgs/sr/candidate_1052.png?raw=true "sr6")

* #### **HR**
![Alt text](imgs/hr/candidate_1052.png?raw=true "hr6")


### EXAMPLE 7

* #### **LR**
![Alt text](imgs/lr/candidate_1062.png?raw=true "lr7")

* #### **SR**
![Alt text](imgs/sr/candidate_1062.png?raw=true "sr7")

* #### **HR**
![Alt text](imgs/hr/candidate_1062.png?raw=true "hr7")


### EXAMPLE 8

* #### **LR**
![Alt text](imgs/lr/candidate_1065.png?raw=true "lr8")

* #### **SR**
![Alt text](imgs/sr/candidate_1065.png?raw=true "sr8")

* #### **HR**
![Alt text](imgs/hr/candidate_1065.png?raw=true "hr8")