# Welcome to DeepLearning of Yiqun Chen

This page is the the deep learning experience of Yiqun Chen, and the page is remaining updating.

# Content

1. [ResNet, BatchNorm and papers reading.](#resnet_batchnorm_and_papers_reading)

  1. [ResNet with experiments.](#resnet_with_experiments)
  
  - [Training strategy.](#the_training_strategy)
  
  - [The result of training loss and evaluating accuracy.](#the_result_of_training_loss_and_evaluating_accuracy)
  
  2. [The Batch Normalization.](#the_batch_normalization)

  3. [The papers reading.](#papers_reading)


# ResNet, BatchNorm and papers reading.

## ResNet with experiments.

The best testing accuracy have reached 93.91% at epoch 1339. The more detailed introduction will be available soon.

### The training strategy.

The learning rate is set to be 0.001 for the first 250 epochs for warm up, and then 0.01 for next 250 epochs, when it get 500 epochs and 750 epochs, I set the learning rate to 0.001 and 0.0001. The code is now [available](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211.py)

### The result of training loss and evaluating accuracy.

![loss](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211_acc.png)

![loss](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211_loss.png)

## The Batch Normalization.

You can view the result by [clicking the link](https://www.overleaf.com/read/kgyxrfttszbp).

## Papers reading

I have read the following papers:

- [Identity Mappings in Deep Residual Networks.](https://arxiv.org/pdf/1603.05027.pdf)

- [Joint Discriminative and Generative Learning for Person Re-identification.](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Joint_Discriminative_and_Generative_Learning_for_Person_Re-Identification_CVPR_2019_paper.pdf)

And the paper below is what I am reading:

- [TensorMask: A Foundation for Dense Object Segmentation.](https://arxiv.org/pdf/1903.12174.pdf)
