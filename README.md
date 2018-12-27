# colorNet-pytorch
A Neural Network For Automatic Image Colorization

This project is a PyTorch version of the [ColorNet](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) issued on SIGGRAPH 2016. Please check out the original website for the details.

## Overview
* Net model
![...](https://github.com/shufanwu/colorNet-pytorch/blob/master/readme%20images/model.png)

* DataSet
[MIT Places205](http://places.csail.mit.edu/user/index.php)  
> Hint: For there are grayscale images in the dataset, I write a script to remove these images

* Development Environment  
Python 3.5.1  
CUDA 8.0  

## Result
I just train this model for 3 epochs while 11 epochs in the paper， so I think it will work better if train it more.

* Good results  
![...](https://github.com/shufanwu/colorNet-pytorch/blob/master/readme%20images/good-result.png)  
* Bad results  
![...](https://github.com/shufanwu/colorNet-pytorch/blob/master/readme%20images/bad-result.png)  
For this network is trained by landscape image database, it's work well for scenery pictures. So if you use this network to color  images of other types, maybe you can't get a satisfying output.

## Todo
Implement with PyTorch1.0

