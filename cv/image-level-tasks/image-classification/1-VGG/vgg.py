import torch
from torch import nn
from common import jax as jax
# from d2l import torch as jax


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                    kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

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

lr, num_epochs, batch_size = 0.05, 10, 128
root = "/data/coding/data/"
train_iter, test_iter = jax.load_data_fashion_mnist(root,batch_size, resize=224)
jax.train_ch6(net, train_iter, test_iter, num_epochs, lr, jax.try_gpu())