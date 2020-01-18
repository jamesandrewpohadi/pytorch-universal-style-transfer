# Universal Style Transfer

A project done by [James Andrew Pohadi](https://github.com/jamesandrewpohadi), [Edric](https://github.com/edricwu) and [Vieri Vincent](https://github.com/vierivincent15)

<p align='center'>
  <img src='samples/sample1.jpg' width='50%'/>
</p>

## Contents
- [Background](#background)
- [Approach used](#approach-used)
- [Requirements](#requirements)
- [Running on Colab](#running-on-colab)
- [Running locally](#running-locally)
- [References and Acknowledgement](#references-and-acknowledgement)

## Background

The problem this project is trying to address is the difficulty of digital content creation, focusing on applying filters, for non art-savvy individuals. Imagine a scenario where an individual is tasked to apply a filter, for example Van Gogh’s ​The Starry Night ​painting, to an image. This task may not be as challenging for art-savvy individuals since they are able to either utilize existing art software such as photoshop to produce the requirement, or paint the image such that it fulfills the requirement. However, this will be a huge challenge for those who are not as adept to the process.

This is where this project, image style transfer model, can come in to help those non art-savvy individuals complete the task through some computing. To be more specific, the project focus on universal image style transfer model, which will be able to apply the style of any style images to any input content images.

## Approach used

The group tried several approaches of neural style transfer from different papers and the approach by *Xun Huang* and *Serge Belongie* give the most visually satisfying style transfer for the project:
[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)

We trained our model with **[COCO val 2014](http://cocodataset.org/#download)** dataset as content images and *train_2* of **[Painter By Number](https://www.kaggle.com/c/painter-by-numbers/data)** dataset as style images

Trained model: [coming soon]

## Requirements
[PyTorch](https://pytorch.org/) >= v1.0.0 and CUDA installed

## Running on Colab

1. Go to https://colab.research.google.com/ and upload Colab.ipynb notebook.

2. Change runtime type to GPU: ```Runtime > Change runtime type```, then choose ```GPU``` in the "Hardware accelerator" field.

3. Upload your kaggle.json file to Colab and accept [Painter By Numbers](https://www.kaggle.com/c/painter-by-numbers/data) competition rules by clicking ```I Understand and Accept``` button in the data section. If you don't have kaggle.json, you can download one by creating new API Token from https://www.kaggle.com/<YOUR_ACCOUNT_NAME>/account in the API section.

4. Run the cells accordingly

## Running locally
- [Setting up](#setting-up)
- [Training](#training)
- [Testing](#testing)

### Setting up

1. Clone this repository. Create ```data/content```, ```data/style```, ```weights```, and ```ckpt``` folders.
```Shell
git clone https://github.com/jamesandrewpohadi/pytorch-universal-style-transfer.git
cd pytorch-universal-style-transfer
mkdir -p data/content data/style weights ckpt
```

2. Download [vgg_normalised.pth](https://transformer-cds.s3-ap-southeast-1.amazonaws.com/vgg_normalised.pth) and [decoder.pth](https://transformer-cds.s3-ap-southeast-1.amazonaws.com/decoder.pth). Place it in the project folder.
```Shell
wget https://transformer-cds.s3-ap-southeast-1.amazonaws.com/vgg_normalised.pth
wget https://transformer-cds.s3-ap-southeast-1.amazonaws.com/decoder.pth
```

3. Download [val2014.zip](http://images.cocodataset.org/zips/val2014.zip). Unzip it and place it in ```./data/content/``` folder.
```Shell
wget http://images.cocodataset.org/zips/val2014.zip
unzip -qq val2014.zip
mv val2014 data/content/
```

4. Download [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers/data) train_2.zip. Extract it, then move the images inside it to ```./data/style/``` folder.
```Shell
mv path/to/train_2 data/style
```

### Training
```Shell
python3 train.py
```

### Testing
To test, put content images in ```./test/content``` and style images in ```./test/style/```, then run following command.
```Shell
python3 test.py
```

## Results

<p float="left">
  <img src="samples/image_4.jpg" width="30%" />
  <img src="samples/batik1.jpg" width="30%" />
  <img src="samples/batik1_12.jpg" width="30%" />
  
  
  
</p>
<p float="left">
  <img src="samples/image_14.jpg" width="30%" /> 
  <img src="samples/painting2.jpg" width="30%" /> 
  <img src="samples/painting2_5.jpg" width="30%" /> 

</p>
<p float="left">
  <img src="samples/image_15.jpg" width="30%" />
  <img src="samples/flower.jpg" width="30%" />
  <img src="samples/flower_23.jpg" width="30%" />
</p>
<p float="left">
  <img src="samples/image_20.jpg" width="30%" />
  <img src="samples/painting4.jpg" width="30%" />
  <img src="samples/painting4_3.jpg" width="30%" />
</p>

## References
- [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
