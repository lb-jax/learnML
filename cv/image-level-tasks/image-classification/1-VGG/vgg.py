import torch
from torch import nn
from common import BaseFunc as jax




'''
1、由n个3x3卷积构成，卷积前后特征图大小保持不变，通道数变为指定的通道数
2、多层卷积完成后通过池化汇聚层将特征图尺寸减半
'''
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                    kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

'''
假设输入为224*224的尺寸，如果每个stage会让尺寸减半，通道数增加一倍
那么VGG和ResNet都有5个stage：224 112 56 28 14 7
'''
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, 
        nn.Flatten(),

        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), 
        nn.ReLU(), 
        nn.Dropout(0.5),

        nn.Linear(4096, 4096), 
        nn.ReLU(), 
        nn.Dropout(0.5),

        nn.Linear(4096, 10)
    )

net = vgg(conv_arch)


def training(net,lr,num_epochs, batch_size):
    root = "/data/coding/data/"
    train_iter, test_iter = jax.load_data_fashion_mnist(batch_size,root, resize=224)
    jax.train_ch6(net, train_iter, test_iter, num_epochs, lr, jax.try_gpu())
    
    return net



lr, num_epochs, batch_size = 0.05, 10, 128

net = training(net,lr,num_epochs, batch_size)


output_path = "/data/coding/learnML/cv/image-level-tasks/image-classification/1-VGG/model/vgg_model.onnx"
net.cpu()
# 导出ONNX
jax.export_to_onnx(
    net,
    output_path,
    input_shape=(1, 1,224, 224)  # VGG的标准输入尺寸
)

'''
loss 0.172, train acc 0.937, test acc 0.922
493.4 examples/sec on cuda:0
'''