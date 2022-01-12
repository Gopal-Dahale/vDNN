#ifndef ALEXNET
#define ALEXNET
#include <vector>

class Alexnet {
 public:
  // VGG NET
  std::vector<LayerSpecifier> layer_specifier;
  Alexnet() {
    {
      ConvDescriptor layer0;
      layer0.initializeValues(3, 96, 11, 11, 227, 227, 0, 0, 4, 4, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = layer0;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor layer1;
      layer1.initializeValues(96, 3, 3, 55, 55, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = layer1;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor layer2;
      layer2.initializeValues(96, 256, 5, 5, 27, 27, 2, 2, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = layer2;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor layer3;
      layer3.initializeValues(256, 3, 3, 27, 27, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = layer3;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor layer4;
      layer4.initializeValues(256, 384, 3, 3, 13, 13, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = layer4;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor layer5;
      layer5.initializeValues(384, 384, 3, 3, 13, 13, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = layer5;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor layer6;
      layer6.initializeValues(384, 256, 3, 3, 13, 13, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = layer6;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor layer7;
      layer7.initializeValues(256, 3, 3, 13, 13, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = layer7;
      layer_specifier.push_back(temp);
    }
    {
      FCDescriptor layer8;
      layer8.initializeValues(9216, 4096, RELU);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = layer8;
      layer_specifier.push_back(temp);
    }
    {
      FCDescriptor layer9;
      layer9.initializeValues(4096, 4096, RELU);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = layer9;
      layer_specifier.push_back(temp);
    }
    {
      FCDescriptor layer10;
      layer10.initializeValues(4096, 1000);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = layer10;
      layer_specifier.push_back(temp);
    }
    {
      SoftmaxDescriptor layer11;
      layer11.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 10, 1,
                               1);
      LayerSpecifier temp;
      temp.initPointer(SOFTMAX);
      *((SoftmaxDescriptor *)temp.params) = layer11;
      layer_specifier.push_back(temp);
    }
  }
};

#endif