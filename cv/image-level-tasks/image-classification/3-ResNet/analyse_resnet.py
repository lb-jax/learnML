import numpy as np

oi_list = []

# 计算卷积层的FLOPs和访存量
def conv_flops_mem(in_shape, out_channels, kernel_size, stride=1, padding=0, bias=True):
    """
    计算卷积层的FLOPs和理论最低访存量
    
    参数:
        in_shape: 输入特征图形状 (C_in, H_in, W_in)
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
    C_in, H_in, W_in = in_shape
    
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
    input_mem = C_in * H_in * W_in * 4
    output_mem = out_channels * H_out * W_out * 4
    weight_mem = out_channels * C_in * kernel_size[0] * kernel_size[1] * 4
    bias_mem = out_channels * 4 if bias else 0
    
    mem_access = input_mem + output_mem + weight_mem + bias_mem

    oi = flops/(mem_access/4)
    global oi_list
    oi_list.append(oi)
    
    return flops, mem_access, out_shape

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
    input_mem = in_features * 4
    output_mem = out_features * 4
    weight_mem = in_features * out_features * 4
    bias_mem = out_features * 4 if bias else 0
    
    mem_access = input_mem + output_mem + weight_mem + bias_mem
    
    oi = flops/(mem_access/4)
    global oi_list
    oi_list.append(oi)
    return flops, mem_access

# 计算池化层的访存量(忽略计算量)
def pool_mem(in_shape, kernel_size, stride=2, padding=0):
    """
    计算池化层的理论最低访存量(忽略计算量)
    
    参数:
        in_shape: 输入特征图形状 (C, H, W)
        kernel_size: 池化核尺寸
        stride: 步长
        padding: 填充
        
    返回:
        mem_access: 理论最低访存量 (bytes)
        out_shape: 输出特征图形状
    """
    C, H_in, W_in = in_shape
    
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
    input_mem = C * H_in * W_in * 4
    output_mem = C * H_out * W_out * 4
    mem_access = input_mem + output_mem
    
    return mem_access, out_shape

# 计算全局平均池化层的访存量
def global_avg_pool_mem(in_shape):
    """
    计算全局平均池化层的理论最低访存量(忽略计算量)
    
    参数:
        in_shape: 输入特征图形状 (C, H, W)
        
    返回:
        mem_access: 理论最低访存量 (bytes)
        out_shape: 输出特征图形状 (C, 1, 1)
    """
    C, H, W = in_shape
    out_shape = (C, 1, 1)
    
    # 全局平均池化: 输入 + 输出
    input_mem = C * H * W * 4
    output_mem = C * 1 * 1 * 4
    mem_access = input_mem + output_mem
    
    return mem_access, out_shape

# ResNet50的Bottleneck模块计算
def bottleneck_flops_mem(in_shape, mid_channels, out_channels, stride=1, downsample=False):
    """
    计算一个Bottleneck模块的FLOPs和访存量
    
    参数:
        in_shape: 输入特征图形状 (C_in, H_in, W_in)
        mid_channels: 中间层通道数
        out_channels: 输出通道数
        stride: 步长
        downsample: 是否下采样（是否需要1x1卷积调整维度）
        
    返回:
        total_flops: 总浮点运算次数
        total_mem: 总访存量 (bytes)
        out_shape: 输出特征图形状
    """
    C_in, H_in, W_in = in_shape
    total_flops = 0
    total_mem = 0
    
    # 主路径的三个卷积层
    # conv1: 1x1, 降维
    flops, mem, shape = conv_flops_mem(in_shape, mid_channels, kernel_size=1, stride=1)

    total_flops += flops
    total_mem += mem
    
    # conv2: 3x3, 特征提取 (可能带下采样)
    flops, mem, shape = conv_flops_mem(shape, mid_channels, kernel_size=3, stride=stride, padding=1)
    total_flops += flops
    total_mem += mem
    
    # conv3: 1x1, 升维
    flops, mem, out_shape = conv_flops_mem(shape, out_channels, kernel_size=1, stride=1)
    total_flops += flops
    total_mem += mem
    
    # 下采样路径（如果需要）
    if downsample or C_in != out_channels:
        # 使用1x1卷积调整通道数或下采样
        flops, mem, _ = conv_flops_mem(in_shape, out_channels, kernel_size=1, stride=stride)
        total_flops += flops
        total_mem += mem
    
    # 残差相加（访存：读两个特征图 + 写一个结果）
    # 假设两个特征图大小相同：out_channels * H_out * W_out
    H_out, W_out = out_shape[1], out_shape[2]
    mem_add = out_channels * H_out * W_out * 4 * 3  # 读A、读B、写结果
    total_mem += mem_add
    
    return total_flops, total_mem, out_shape

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
    # Conv7x7 s=2, pad=3
    flops, mem, shape = conv_flops_mem(input_shape, 64, kernel_size=7, stride=2, padding=3)
    part_results['initial']['flops'] += flops
    part_results['initial']['mem'] += mem
    
    # MaxPool k=3, s=2
    mem, shape = pool_mem(shape, kernel_size=3, stride=2, padding=1)
    part_results['initial']['mem'] += mem
    
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
        stage, num_blocks, in_channels, mid_channels, out_channels, first_stride = config
        
        # 每个stage的第一个bottleneck需要下采样
        flops, mem, shape = bottleneck_flops_mem(
            (in_channels, *shape[1:]), 
            mid_channels, 
            out_channels,
            stride=first_stride,
            downsample=True
        )
        part_results['stages']['flops'] += flops
        part_results['stages']['mem'] += mem
        
        # 该stage剩余的bottlenecks
        for _ in range(1, num_blocks):
            flops, mem, shape = bottleneck_flops_mem(
                shape, 
                mid_channels, 
                out_channels,
                stride=1,
                downsample=False
            )
            part_results['stages']['flops'] += flops
            part_results['stages']['mem'] += mem
    
    ##############################
    # 第三部分: 分类头
    ##############################
    # 全局平均池化
    mem, shape = global_avg_pool_mem(shape)
    part_results['head']['mem'] += mem
    
    # 全连接层 (2048 -> 1000)
    flops, mem = fc_flops_mem(2048, 1000)
    part_results['head']['flops'] += flops
    part_results['head']['mem'] += mem
    
    # 汇总结果
    total_flops = (part_results['initial']['flops'] + 
                   part_results['stages']['flops'] + 
                   part_results['head']['flops'])
    
    total_mem = (part_results['initial']['mem'] + 
                 part_results['stages']['mem'] + 
                 part_results['head']['mem'])
    
    return total_flops, total_mem, part_results

# 主函数
if __name__ == "__main__":
    total_flops, total_mem, parts = resnet50_flops_mem()

    
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
    print(f"\nCompute-Memory Ratio (FLOPs/byte): {flops_mem_ratio:.4f}")

    oi_list.sort()
    for oi in oi_list:
        print(oi)