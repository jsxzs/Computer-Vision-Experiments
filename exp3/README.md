## Exp3: Compress Deep Neural Network With Pruning

Please refer to [exp3 requirements](./%E4%B8%8A%E6%9C%BA%E5%AE%9E%E9%AA%8C%E4%B8%89.docx) for more details.

Download work_dir [here](https://drive.google.com/drive/folders/1O2jsRZCIPPoS6uzmSbIFIsVM_SGvNybJ?usp=share_link), which contains training logs, json files and .pth files.

In this experiment, I only use ResNet9 and VGG13_bn trained in [exp2](../exp2/).

Folder structure is the same as exp2.

Results:
- ResNet9
  The feature maps of the last conv layer:

  ![](../images/exp3/resnet9_featuremaps.png)

  Test accuracy change with respect to pruned neuron numbers:

  ![](../images/exp3/resnet9_curve.png)

- VGG13_bn
  The feature maps of the last conv layer:

  ![](../images/exp3/vgg13bn_featuremaps.png)

  Test accuracy change with respect to pruned neuron numbers:

  ![](../images/exp3/vgg13bn_curve.png)

Analysis is in the [exp3_report](../exp3/report.pdf).

