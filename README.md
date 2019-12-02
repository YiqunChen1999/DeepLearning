# Welcome to DeepLearning of Yiqun Chen

This page is the the deep learning experience of Yiqun Chen, and the page is remaining updating.

# Content

1. [ResNet, BatchNorm and papers reading.](#resnet-batchnorm-papers_reading)

    1.1. [ResNet with experiments.](#resnet_with_experiments)
  
    1.2. [The Batch Normalization.](#batchnorm)

    1.3. [The papers reading.](#papers_reading)
  
2. [Other Resource](#other_resource)


# ResNet, BatchNorm and papers reading. {#resnet-batchnorm-papers_reading}

## ResNet with experiments. {#resnet_with_experiments}

The best testing accuracy have reached 93.91% of res50 version 211 and 94.01% of res50 version 212. The more detailed introduction will be available soon.

### The training strategy. {#training_strategy}

#### res50_v211

The learning rate is set to be 0.001 for the first 250 epochs for warm up, and then 0.01 for next 250 epochs, when it get 500 epochs and 750 epochs, I set the learning rate to 0.001 and 0.0001. The code is now [available](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211.py)

#### res50_v212

The learning rate is first set to be 0.001 for the first 500 epochs for warm up, when it reach epoch 500, 900, 1200, the learning rate is set to be 0.01, 0.001, 0.0001 respectively. The code is now [available](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v212/res50_v212.py)

### The result of training loss and evaluating accuracy. {#res50_training_result}

#### res50_v211

![accuracy](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211_acc.png)

![loss](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211_loss.png)

#### res50_v212

![accuracy](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v212/res50_v212_acc.png)

![loss](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v212/res50_v212_loss.png)


## The Batch Normalization. {#batchnorm}

You can view the result by [clicking the link](https://www.overleaf.com/read/kgyxrfttszbp).

## Papers reading. {#papers_reading}

I have read the following papers:

- [Identity Mappings in Deep Residual Networks.](https://arxiv.org/pdf/1603.05027.pdf)

- [Joint Discriminative and Generative Learning for Person Re-identification.](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Joint_Discriminative_and_Generative_Learning_for_Person_Re-Identification_CVPR_2019_paper.pdf)


# Other Resource

[Deep Learning for Tracking and Detection](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection)

[Deep Learning for Object Detection](https://github.com/hoya012/deep_learning_object_detection)

[Awesome Graph Classification](https://github.com/benedekrozemberczki/awesome-graph-classification)

[Awesome Object Detection](https://github.com/amusi/awesome-object-detection)

[Pumpkin Book](https://github.com/datawhalechina/pumpkin-book)
