# RCNN_TF2

Implementation part of Deep learning course on Recurrent Convolutional Neural Networks (RCNN) for object classification.
The project aims to optimize two major hyper-parameters in the architecture of the proposed RCNN by Liang and Hu [1] -
the number of RCL blocks in the network and the number of convolutions within each block.

The project consists of a single python file 'main.py' that should by run using python without further arguments.
When running, a folder 'hparams_tuning' is created for each run in hyper-parameter space (num of RCL blocks, num of convolutions in each RCL block).
In the end of each run, another folder 'pre-trained' is created with the trained network of a specific tuple of hyper-parameters. 


[1] Ming Liang and Xiaolin Hu, "Recurrent convolutional neural network for object recognition",
in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015), pp. 3367-3375.
