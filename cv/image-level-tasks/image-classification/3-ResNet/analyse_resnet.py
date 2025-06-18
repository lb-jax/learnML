import numpy as np
import copy
from collections import OrderedDict



DATE_TYPE__BYTES = 1

# 计算卷积层的FLOPs和访存量
def conv_flops_mem(shape, out_channels, kernel_size, stride=1, padding=0, bias=True):
    """
    计算卷积层的FLOPs和理论最低访存量
    
    参数:
        shape: 输入特征图形状 (C_in, H_in, W_in)
        out_channels: 输出通道数
        kernel_size: 卷积核尺寸
        stride: 步长
        padding: 填充
        bias: 是否使用偏置
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        out_shape: 输出特征图形状 (C_out, H_out, W_out)
    """
    C_in, H_in, W_in = shape
    
    # 计算输出特征图尺寸
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    H_out = (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    out_shape = (out_channels, H_out, W_out)
    
    # 计算FLOPs (乘加操作算2次浮点运算: 1次乘法+1次加法)
    flops_per_output = C_in * kernel_size[0] * kernel_size[1]
    flops = out_channels * H_out * W_out * flops_per_output * 2
    
    if bias:
        flops += out_channels * H_out * W_out * 2  # 偏置加法
    
    # 计算理论最低访存量 (单位: bytes, float32=4 bytes)
    # 输入特征图 + 输出特征图 + 权重 + 偏置(如果有)
    input_mem = C_in * H_in * W_in * DATE_TYPE__BYTES
    output_mem = out_channels * H_out * W_out * DATE_TYPE__BYTES
    weight_mem = out_channels * C_in * kernel_size[0] * kernel_size[1] * DATE_TYPE__BYTES
    bias_mem = out_channels * DATE_TYPE__BYTES if bias else 0
    
    mem_access = input_mem + output_mem + weight_mem + bias_mem  
    return flops, mem_access, out_shape

# 计算ReLU层的FLOPs和访存量
def relu_flops_mem(input_shape):
    """
    计算ReLU激活函数的FLOPs和访存量
    参数:
        input_shape: 输入形状 (C, H, W)
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (与输入相同)
    """
    C, H, W = input_shape
    num_elements = C * H * W
    
    # ReLU的计算量：每个元素比较1次（最大值函数）
    flops = num_elements
    
    # 访存量: 输入 + 输出 (相同形状)
    mem_access = 2 * num_elements * DATE_TYPE__BYTES
    
    return flops, mem_access, input_shape

# 计算BN层的FLOPs和访存量
def bn_flops_mem(input_shape, affine=True):
    """
    计算批归一化层(BN)的FLOPs和访存量
    参数:
        input_shape: 输入形状 (C, H, W)
        affine: 是否包含可学习参数
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (与输入相同)
    """
    C, H, W = input_shape
    num_elements = C * H * W
    
    # BN计算:
    # 1. 归一化: (x - mean)/std
    # 2. 缩放和偏移: γ*x_norm + β
    flops_per_element = 4  # 减、除、乘、加
    flops = num_elements * flops_per_element
    
    # 访存量:
    # 输入特征图 + 输出特征图
    mem_access = 2 * num_elements * DATE_TYPE__BYTES
    
    # 增加可学习参数γ和β的访问
    if affine:
        mem_access += 2 * C * DATE_TYPE__BYTES  # γ和β各C个元素
    
    return flops, mem_access, input_shape

# 计算全连接层的FLOPs和访存量
def fc_flops_mem(in_features, out_features, bias=True):
    """
    计算全连接层的FLOPs和理论最低访存量
    
    参数:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
    """
    # 计算FLOPs
    flops = in_features * out_features * 2  # 乘加操作
    if bias:
        flops += out_features * 2  # 偏置加法
    
    # 计算理论最低访存量
    input_mem = in_features * DATE_TYPE__BYTES
    output_mem = out_features * DATE_TYPE__BYTES
    weight_mem = in_features * out_features * DATE_TYPE__BYTES
    bias_mem = out_features * DATE_TYPE__BYTES if bias else 0
    
    mem_access = input_mem + output_mem + weight_mem + bias_mem
    

    return flops, mem_access

# 计算池化层的访存量(忽略计算量)
def pool_mem(shape, kernel_size, stride=2, padding=0):
    """
    计算池化层的理论最低访存量(忽略计算量)
    
    参数:
        shape: 输入特征图形状 (C, H, W)
        kernel_size: 池化核尺寸
        stride: 步长
        padding: 填充
        
    返回:
        mem_access: 理论最低访存量 (bytes)
        out_shape: 输出特征图形状
    """
    C, H_in, W_in = shape
    
    # 计算输出特征图尺寸
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    H_out = (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    out_shape = (C, H_out, W_out)
    
    # 池化层只有输入和输出的访存
    input_mem = C * H_in * W_in * DATE_TYPE__BYTES
    output_mem = C * H_out * W_out * DATE_TYPE__BYTES
    mem_access = input_mem + output_mem
    
    return mem_access, out_shape

# 计算全局平均池化层的访存量
def global_avg_pool_mem(shape):
    """
    计算全局平均池化层的理论最低访存量(忽略计算量)
    
    参数:
        shape: 输入特征图形状 (C, H, W)
        
    返回:
        mem_access: 理论最低访存量 (bytes)
        out_shape: 输出特征图形状 (C, 1, 1)
    """
    C, H, W = shape
    out_shape = (C, 1, 1)
    
    # 全局平均池化: 输入 + 输出
    input_mem = C * H * W * DATE_TYPE__BYTES
    output_mem = C * 1 * 1 * DATE_TYPE__BYTES
    mem_access = input_mem + output_mem
    
    return mem_access, out_shape

# ResNet50的Bottleneck模块计算
def bottleneck_flops_mem(shape, mid_channels, out_channels, stride=1, downsample=False):
    """
    计算一个Bottleneck模块的FLOPs和访存量
    
    参数:
        shape: 输入特征图形状 (C_in, H_in, W_in)
        mid_channels: 中间层通道数
        out_channels: 输出通道数
        stride: 步长
        downsample: 是否下采样（是否需要1x1卷积调整维度）
        
    返回:
        total_flops: 总浮点运算次数
        total_mem: 总访存量 (bytes)
        out_shape: 输出特征图形状
    """
    bottleneck_record = {}
    bottleneck_layer_1 = {}
    bottleneck_layer_2 = {}
    bottleneck_layer_3 = {}

    C_in, H_in, W_in = shape
    total_flops = 0
    total_mem = 0
    
    # 主路径的三个卷积层
    # conv1: 1x1, 降维
    # 第一层：1x1卷积 -> BN -> ReLU
    # conv1: 1x1, stride=1, 降维 (带BN和ReLU)
    in_shape = shape
    flops, mem, shape = conv_flops_mem(shape, mid_channels, kernel_size=1, stride=1)
    total_flops += flops
    total_mem += mem
    layer_1_conv1x1 = {
        "In_Shape":in_shape,
        "Weight_Shape":[shape[0],in_shape[0],1,1],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = bn_flops_mem(shape)
    total_flops += flops
    total_mem += mem
    layer_1_bn = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = relu_flops_mem(shape)
    total_flops += flops
    total_mem += mem
    layer_1_relu = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }
    bottleneck_layer_1["layer_1_conv1x1"] = layer_1_conv1x1
    bottleneck_layer_1["layer_1_bn"] = layer_1_bn
    bottleneck_layer_1["layer_1_relu"] = layer_1_relu

    
    # 第二层：3x3卷积 -> BN -> ReLU
    # conv2: 3x3, 可能stride=2 (带BN和ReLU)
    in_shape = shape
    flops, mem, shape = conv_flops_mem(shape, mid_channels, kernel_size=3, stride=stride, padding=1)
    total_flops += flops
    total_mem += mem
    layer_2_conv3x3 = {
        "In_Shape":in_shape,
        "Weight_Shape":[shape[0],in_shape[0],3,3],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = bn_flops_mem(shape)
    total_flops += flops
    total_mem += mem
    layer_2_bn = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = relu_flops_mem(shape)
    total_flops += flops
    total_mem += mem
    layer_2_relu = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }
    bottleneck_layer_2["layer_2_conv3x3"] = layer_2_conv3x3
    bottleneck_layer_2["layer_2_bn"] = layer_2_bn
    bottleneck_layer_2["layer_2_relu"] = layer_2_relu

    

    # 第三层：1x1卷积 -> BN (不带ReLU)
    # conv3: 1x1, 升维 (只有BN，不含ReLU)
    in_shape = shape
    flops, mem, out_shape = conv_flops_mem(shape, out_channels, kernel_size=1, stride=1)
    total_flops += flops
    total_mem += mem
    layer_3_conv1x1 = {
        "In_Shape":in_shape,
        "Weight_Shape":[shape[0],in_shape[0],1,1],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = bn_flops_mem(shape)
    total_flops += flops
    total_mem += mem
    layer_3_bn = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }
    bottleneck_layer_3["layer_3_conv1x1"] = layer_3_conv1x1
    bottleneck_layer_3["layer_3_bn"] = layer_3_bn
    
    # 在shortcut中，下采样（如果需要）
    if downsample or C_in != out_channels:
        # 使用1x1卷积调整通道数或下采样
        in_shape = shape
        flops, mem, shape = conv_flops_mem(shape, out_channels, kernel_size=1, stride=stride)
        total_flops += flops
        total_mem += mem
        layer_3_shortcut = {
            "In_Shape":in_shape,
            "Weight_Shape":[shape[0],in_shape[0],1,1],
            "Out_Shape":shape,
            "Flops":flops,
            "Mem":mem,
            "Type":"CIMM",
            "Arithmetic_Intensity":flops/mem
        }
        bottleneck_layer_3["layer_3_shortcut"] = layer_3_shortcut
    
    # 残差相加（访存：读两个特征图 + 写一个结果）
    # 假设两个特征图大小相同：out_channels * H_out * W_out
    H_out, W_out = out_shape[1], out_shape[2]
    flops = out_channels * H_out * W_out * DATE_TYPE__BYTES
    mem_add = out_channels * H_out * W_out * DATE_TYPE__BYTES * 3  # 读A、读B、写结果
    total_flops += flops
    total_mem += mem_add
    layer_3_add = {
        "In_Shape":out_shape,
        "Weight_Shape":[0],
        "Out_Shape":out_shape,
        "Flops":flops,
        "Mem":mem_add,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem_add
    }
    bottleneck_layer_3["layer_3_add"] = layer_3_add

    bottleneck_record["bottleneck_layer_1"] = bottleneck_layer_1
    bottleneck_record["bottleneck_layer_2"] = bottleneck_layer_2
    bottleneck_record["bottleneck_layer_3"] = bottleneck_layer_3
    
    return total_flops, total_mem, out_shape,bottleneck_record

# 计算整个ResNet50的FLOPs和访存量
def resnet50_flops_mem():
    # 初始化
    total_flops = 0
    total_mem = 0
    
    # 用于存储各部分结果
    part_results = {
        'initial': {'flops': 0, 'mem': 0},
        'stages': {'flops': 0, 'mem': 0},
        'head': {'flops': 0, 'mem': 0}
    }
    
    # 输入尺寸 (C, H, W)
    input_shape = (3, 224, 224)
    
    ##############################
    # 第一部分: 初始卷积与池化
    ##############################
    part_1 = {}
    part_2 = {}
    part_3 = {}

    # conv7x7 s=2, pad=3 BN->Relu
    flops, mem, shape = conv_flops_mem(input_shape, 64, kernel_size=7, stride=2, padding=3)
    part_results['initial']['flops'] += flops
    part_results['initial']['mem'] += mem
    part1_conv7x7 = {
        "In_Shape":input_shape,
        "Weight_Shape":[shape[0],input_shape[0],7,7],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = bn_flops_mem(shape)
    part_results['initial']['flops'] += flops
    part_results['initial']['mem'] += mem
    part1_conv7x7_bn = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = relu_flops_mem(shape)
    part_results['initial']['flops'] += flops
    part_results['initial']['mem'] += mem
    part1_conv7x7_relu = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }
    
     
    
    # MaxPool k=3, s=2;
    in_shape = shape
    mem, shape = pool_mem(shape, kernel_size=3, stride=2, padding=1)
    part_results['initial']['mem'] += mem
    part1_MaxPool = {
        "In_Shape":in_shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = bn_flops_mem(shape)
    part_results['initial']['flops'] += flops
    part_results['initial']['mem'] += mem
    part1_MaxPool_bn = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    flops, mem, shape = relu_flops_mem(shape)
    part_results['initial']['flops'] += flops
    part_results['initial']['mem'] += mem
    part1_MaxPool_relu = {
        "In_Shape":shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }

    part_1["part1_conv7x7"] = part1_conv7x7
    part_1["part1_conv7x7_bn"] = part1_conv7x7_bn
    part_1["part1_conv7x7_relu"] = part1_conv7x7_relu
    part_1["part1_MaxPool"] = part1_MaxPool
    part_1["part1_MaxPool_bn"] = part1_MaxPool_bn
    part_1["part1_MaxPool_relu"] = part1_MaxPool_relu
    
    ##############################
    # 第二部分: Stage1-Stage4
    ##############################
    # ResNet50各阶段配置: 
    # (stage, num_blocks, in_channels, mid_channels, out_channels, stride)
    # stride只在每个stage的第一个bottleneck使用
    stages_config = [
        (1, 3, 64, 64, 256, 1),   # Stage1
        (2, 4, 256, 128, 512, 2),  # Stage2
        (3, 6, 512, 256, 1024, 2), # Stage3
        (4, 3, 1024, 512, 2048, 2) # Stage4
    ]
    
 
    
    for config in stages_config:
        stage_record = {}
        stage, num_blocks, in_channels, mid_channels, out_channels, first_stride = config
        
        # 每个stage的第一个bottleneck需要下采样
        flops, mem, shape ,record= bottleneck_flops_mem(
            (in_channels, *shape[1:]), 
            mid_channels, 
            out_channels,
            stride=first_stride,
            downsample=True
        )
        stage_record["bottleneck_1"]= copy.deepcopy(record)

        part_results['stages']['flops'] += flops
        part_results['stages']['mem'] += mem
        
        # 该stage剩余的 bottlenecks
        for i in range(1, num_blocks):
            flops, mem, shape,bottleneck_record = bottleneck_flops_mem(
                shape, 
                mid_channels, 
                out_channels,
                stride=1,
                downsample=False
            )
            key = "bottlenecks_" + str(i+1)
            stage_record[key] = copy.deepcopy(bottleneck_record)
            part_results['stages']['flops'] += flops
            part_results['stages']['mem'] += mem
        
        key = "stage_" +  str(stage)
        part_2[key] = copy.deepcopy(stage_record)
    
    
    ##############################
    # 第三部分: 分类头
    ##############################
    # 全局平均池化
    in_shape = shape
    mem, shape = global_avg_pool_mem(shape)
    part_results['head']['mem'] += mem
    part_3_global_avg_pool = {
        "In_Shape":in_shape,
        "Weight_Shape":[0],
        "Out_Shape":shape,
        "Flops":0,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":0
    }
    
    # 全连接层 (2048 -> 1000)
    flops, mem = fc_flops_mem(2048, 1000)
    part_results['head']['flops'] += flops
    part_results['head']['mem'] += mem
    part_3_fc = {
        "In_Shape":shape,
        "Weight_Shape":[1000,2048],
        "Out_Shape":(1000),
        "Flops":flops,
        "Mem":mem,
        "Type":"CIMM",
        "Arithmetic_Intensity":flops/mem
    }
    part_3["part_3_global_avg_pool"] = part_3_global_avg_pool
    part_3["part_3_fc"] = part_3_fc
    
    # 汇总结果
    total_flops = (part_results['initial']['flops'] + 
                   part_results['stages']['flops'] + 
                   part_results['head']['flops'])
    
    total_mem = (part_results['initial']['mem'] + 
                 part_results['stages']['mem'] + 
                 part_results['head']['mem'])
    
    total_record = {
        "part_1":part_1,
        "part_2":part_2,
        "part_3":part_3
    }
    
    return total_flops, total_mem, part_results,total_record

def extract_arithmetic_intensity(data, sort=True, reverse=False):
    # 递归提取结果的普通字典
    result_dict = {}
    
    def traverse(current_dict, path_keys):
        has_nested_dict = False
        for key, value in current_dict.items():
            if isinstance(value, dict):
                has_nested_dict = True
                new_path = path_keys + [key]
                traverse(value, new_path)
        if not has_nested_dict:
            if "Arithmetic_Intensity" in current_dict:
                full_path = path_keys + ["Arithmetic_Intensity"]
                key_name = "_".join(full_path)
                result_dict[key_name] = current_dict["Arithmetic_Intensity"]
    
    traverse(data, [])
    
    # 是否按值排序
    if sort:
        # 将字典项按值排序（浮点数），这里用sorted，key为值
        sorted_items = sorted(result_dict.items(), key=lambda x: x[1], reverse=reverse)
        # 构造有序字典
        return OrderedDict(sorted_items)
    else:
        return result_dict

import json

def save_dict_to_json(data, file_path, indent=4):
    """
    将嵌套字典保存为 JSON 文件
    
    参数:
    data -- 要保存的嵌套字典
    file_path -- 要保存的文件路径（如 'data.json'）
    indent -- JSON 文件缩进（美化格式），默认 4 个空格
    """
    try:
        # 打开文件并写入 JSON 数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        print(f"字典已成功保存到 '{file_path}'")
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False

# 主函数
if __name__ == "__main__":
    total_flops, total_mem, parts,total_record = resnet50_flops_mem()

    
    # 打印结果
    print("ResNet50 FLOPs and Memory Access Breakdown:")
    print(f"{'Part':<15} | {'FLOPs (G)':>15} | {'Mem Access (MB)':>15}")
    print("-" * 50)
    
    # 格式化各部分数据
    def format_flops(flops):
        return f"{flops / 1e9:.2f}"
    
    def format_mem(mem):
        return f"{(mem / 4) / (1024**2):.2f}"
    
    print(f"{'Initial':<15} | {format_flops(parts['initial']['flops']):>15} | {format_mem(parts['initial']['mem']):>15}")
    print(f"{'Stages':<15} | {format_flops(parts['stages']['flops']):>15} | {format_mem(parts['stages']['mem']):>15}")
    print(f"{'Head':<15} | {format_flops(parts['head']['flops']):>15} | {format_mem(parts['head']['mem']):>15}")
    print("-" * 50)
    print(f"{'Total':<15} | {format_flops(total_flops):>15} | {format_mem(total_mem):>15}")

    flops_mem_ratio = total_flops / (total_mem/4)  # FLOPs per byte
    print(f"\nCompute-Memory Ratio (TLOPs/byte): {flops_mem_ratio:.4f}")

    original_record_parh = "./original_record.json"
    save_dict_to_json(total_record,original_record_parh)

    
    record = extract_arithmetic_intensity(total_record)
    record_parh = "./record.json"
    save_dict_to_json(record,record_parh)

