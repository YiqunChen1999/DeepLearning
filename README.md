# Welcome to DeepLearning of Yiqun Chen

This page is the the deep learning experience of Yiqun Chen.

# Content

1. ResNet, BatchNorm and papers reading.

  1. ResNet with experiments.
  
  - Training strategy.
  
  - The result of training loss and evaluating accuracy.
  
  2. The Batch Normalization.

  3. The papers reading.


# ResNet, BatchNorm and papers reading.

## The training strategy.

The learning rate is set to be 0.001 for the first 250 epochs for warm up, and then 0.01 for next 250 epochs, when it get 500 epochs and 750 epochs, I set the learning rate to 0.001 and 0.0001. The code is available at (https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211.py)

## The result of training loss and evaluating accuracy.

![loss](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211_acc.png)

![loss](https://raw.githubusercontent.com/YiqunChen1999/DeepLearning/master/res50_v211/res50_v211_loss.png)

