import numpy as np
from typing import Tuple, Dict, List, Any, Union
import numpy as np
import copy
from collections import OrderedDict
import json
from enum import Enum

DATE_TYPE__BYTES = 1

'''
算子融合级别，后续可以结合硬件配置来考虑哪些计算可以融合
'''
class FUSION_LEVEL(Enum):
    NONE = 1
    OPERATOR = 2
    BOTTLENECK = 3
    STAGE = 4 


'''
1. 规则计算密集型 + 规则访存密集型: CIMM -(Compute-Intensive Matrix Multiplication);
2. 规则访存密集型 + 相对轻量计算: MEML - (MEMory + Light Compute);
3. 规则访存 + 归约密集型 (多步访存): REDM - (REDuction + Multiple memory passes);
4. 规则计算/访存但有依赖性或序列性: SEQD - (SEQential Dependency);
5. 不规则/随机访存密集型 + 计算通常相对轻量: IRMA - (IRregular Memory Access);
6. 动态计算与控制密集型 + 潜在不规则访存: DYNC - (DYnamic Control); 
7. 特殊函数与变换: SPEC - (SPECial functions);
'''
class ARITHMTIC_INTENSITY_TYPE(Enum):
    CIMM = 1
    MEML = 2
    REDM = 3
    SEQD = 4
    IRMA = 5
    DYNC = 6
    SPEC = 7


'''
用于保存每个计算块的信息
'''
class CalculateBlockInfo:
    
    def __init__(self,in_mem:int,w_mem:int,out_mem:int,ops:int,name:int,
                 in_shape=[],out_shape=[],w_shape=[],shape=[],
                 type:ARITHMTIC_INTENSITY_TYPE=ARITHMTIC_INTENSITY_TYPE.CIMM):
        self.in_mem = in_mem
        self.w_mem = w_mem
        self.out_mem = out_mem
        self.mem = self.in_mem + self.w_mem + self.out_mem
        self.ops = ops
        self.name = name
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.w_shape = w_shape
        self.type = type
        self.arithmetic_intensity = self.ops / self.mem
        self.shape = shape
    
    def to_dict(self) -> dict:
        """将对象转换为字典形式"""
        return {
            "In_Shape":self.in_shape,
            "Weight_Shape":self.w_shape,
            "Out_Shape":self.out_shape,
            "ops": self.ops,
            "Mem": self.mem,
            "Calculate_Type": self.calculate_type,
            "Arithmetic_Intensity": self.arithmetic_intensity
        }
    
    def __repr__(self) -> str:
        """对象的字符串表示"""
        return (f"CalculateBlockInfo(block_name={self.block_name},ops={self.ops}, mem={self.mem}, "
                f"ct='{self.calculate_type}', ai={self.arithmetic_intensity})")
    

def get_flops_and_memory_access(op_type: str, 
                                inputs: List[Union[np.ndarray, int, float]], 
                                attributes: Dict[str, Any] = None) -> CalculateBlockInfo:
    """
    计算指定ONNX算子的FLOPs和内存访问量
    
    :param op_type: 算子类型（字符串）
    :param inputs: 输入张量或参数的列表
    :param attributes: 算子属性字典
    :return: (FLOPs, (特征输入访问量, 权重输入访问量, 输出访问量))
    """
    if attributes is None:
        attributes = {}
    
    # 根据算子类型调用相应的计算函数
    if op_type == 'Conv':
        return _conv_flops(inputs, attributes)
    elif op_type == 'MatMul':
        return _matmul_flops(inputs, {})
    elif op_type == 'Gemm':
        return _matmul_flops(inputs, attributes)  # Gemm有额外属性
    elif op_type in ['Add', 'Sub', 'Mul', 'Div', 'Pow', 'Mod', 'Equal']:
        return _elementwise_flops(inputs, op_type,attributes)
    elif op_type in ['Relu', 'Sigmoid', 'Softmax', 'Erf', 'Not']:
        return _activation_flops(inputs, op_type,attributes)
    elif op_type in ['Sin', 'Cos', 'Sqrt']:
        return _math_flops(inputs, op_type,attributes)
    elif op_type in ['Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose', 'Identity', 'Cast']:
        return _shape_change_flops(inputs, op_type, attributes)
    elif op_type in ['Concat', 'Split', 'Slice', 'Gather', 'Tile', 'Expand', 'Pad', 'CumSum']:
        return _sequence_flops(inputs, attributes, op_type)
    elif op_type in ['MaxPool', 'GlobalAveragePool', 'ReduceMean', 'ReduceMax', 'Clip']:
        return _reduction_flops(inputs, attributes, op_type)
    elif op_type == 'Resize':
        return _resize_flops(inputs, attributes)
    elif op_type == 'Where':
        return _where_flops(inputs)
    elif op_type in ['Constant', 'ConstantOfShape']:
        return _constant_flops(inputs, attributes)
    elif op_type in ['Range', 'Shape']:
        return _tensor_gen_flops(inputs, op_type)
    else:
        raise ValueError(f"Unsupported operator type: {op_type}")



# --------------------------------------------------------------
# 核心算子实现
# inputs = [[X],[W],[B]]
# --------------------------------------------------------------

def _conv_flops(input_shape: List, attributes: Dict) -> CalculateBlockInfo:
    """
    Conv算子
    ONNX定义: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
    输入: [X, W, B(可选)]
    """
    # print("--"*50)
    # input_shape = []
    # for name, data in inputs.items():
    #     if data is not None:
    #         input_shape.append(data.shape)
    #         # print(f"  {name}: shape={data.shape}, dtype={data.dtype}")
    #     else:
    #         print(f"  {name}: default")
    #         return 0

    X = input_shape[0]
    W = input_shape[1]
    B = input_shape[2] if len(input_shape) >= 3 else None
    
    # 解析属性
    '''
    Attributes:
        dilations = [1, 1]
        group = 1
        kernel_shape = [7, 7]
        pads = [3, 3, 3, 3]
        strides = [2, 2]
    '''
    dilations = attributes.get('dilations', [1, 1])
    groups = attributes.get('group', 1)
    pads = attributes.get('pads', [0, 0, 0, 0])
    strides = attributes.get('strides', [1, 1])
    
    # 输入输出尺寸
    N, C_in, H_in, W_in = X
    C_out, C_in_g, H_k, W_k = W
    H_out = (H_in + pads[0] + pads[2] - dilations[0] * (H_k - 1) - 1) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - dilations[1] * (W_k - 1) - 1) // strides[1] + 1
    
    # 计算FLOPs
    kernel_ops = H_k * W_k * (C_in // groups)
    flops = (2 * kernel_ops * C_out * H_out * W_out * N) // groups
    if B is not None:
        flops += N * C_out * H_out * W_out  # 偏置加法
    
    # 内存访问
    in_mem = N * C_in * H_in * W_in
    w_mem = C_out * C_in_g * H_k * W_k 
    if B is not None:
        w_mem += C_out
    out_mem = N * C_out * H_out * W_out

    L = H_out * W_out
    M = H_k * H_k * C_in
    N = C_out

    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"conv",X,[1,C_out,W_out,H_out],
                                        W,[L,M,N],ARITHMTIC_INTENSITY_TYPE.CIMM)
    
    return calculate_block


def _matmul_flops(input_shape: List, attributes: Dict) -> Tuple[float, Tuple[float, float, float]]:
    """
    MatMul/Gemm矩阵乘法
    """
    shapeA = input_shape[0]
    shapeB = input_shape[1]
    # 处理转置属性
    transA = attributes.get('transA', 0)
    transB = attributes.get('transB', 0)
    
    alpha = attributes.get('alpha', 1.0)
    beta = attributes.get('beta', 1.0)
    
    A_size = 1
    B_size = 1
    for i in shapeA:
        A_size *= i
    for i in shapeB:
        B_size *= i
    
    # 应用转置
    if transA:
        M, K = shapeA[-1], shapeA[-2]
    else:
        if len(shapeA) >  1:
            M, K = shapeA[-2], shapeA[-1]
        else:
            M, K = shapeA[0],shapeA[0]
        
    if transB:
        if len(shapeB) >  1:
            N, K2 = shapeB[-2], shapeB[-1]
        else:
            s = shapeB[0]
            N, K2 = shapeB[0],shapeB[0]
        # N, K2 = shapeB[-2], shapeB[-1]
    else:
        N, K2 = shapeB[-1], shapeB[-2]
    
    # 验证矩阵乘法合法性
    if K != K2:
        return 0
        # raise ValueError(f"Matrix dimensions incompatible: A({M}x{K}) * B({K2}x{N})")

    # 计算FLOPs (2*M*N*K)
    flops = 2 * M * N * K
    
    # 处理Gemm的alpha和beta
    if alpha != 1.0:
        flops += M * N  # alpha缩放
    
    if beta != 0.0 and len(input_shape) > 2:
        shapeC = input_shape[2]
        C_size = 1
        for i in shapeC:
            C_size *= i
        flops += M * N  # beta缩放
        feature_access = A_size + B_size + C_size
    else:
        feature_access = A_size + B_size

    # 内存访问
    in_mem = feature_access
    w_mem = 0       # 矩阵乘法没有权重，都是特征
    out_mem = M * N


    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"matmul",shapeA,[],
                                        shapeA,[M,N,K],ARITHMTIC_INTENSITY_TYPE.CIMM)
    
    
    return calculate_block

def _elementwise_flops(input_shape: List, op_type: str,attributes: Dict) -> Tuple[float, Tuple[float, float, float]]:
    """
    元素级算子: Add, Sub, Mul, Div, Pow, Mod, Equal
    """
    # 获取广播后的输出尺寸
    # N, C_in, H_in, W_in = inputs[0]
    # shapes = [inp for inp in inputs]
    shapes = input_shape
    if len(shapes) == 1:
        output_shape = shapes[0]
    else:
        # 手动广播规则: 从右侧对齐，取每个维度的最大值
        max_dims = max(len(shape) for shape in shapes)
        padded_shapes = [list(shape) for shape in shapes]
        for shape in padded_shapes:
            while len(shape) < max_dims:
                shape.insert(0, 1)
        
        output_dims = []
        for dims in zip(*padded_shapes):
            output_dims.append(max(dims))
        output_shape = tuple(output_dims)
    
    output_elements = 1
    for i in output_shape:
        output_elements *= i
    
    # 不同操作的FLOPs因子
    flops_factors = {
        'Add': 1,
        'Sub': 1,
        'Mul': 1,
        'Div': 4,  # 除法成本更高
        'Pow': 8,  # 幂运算成本高
        'Mod': 5,  # 取模成本高
        'Equal': 1,
    }
    
    flops = output_elements * flops_factors.get(op_type, 1)
    

    # 内存访问
    in_mem = output_elements
    w_mem = 0       # 矩阵乘法没有权重，都是特征
    out_mem = output_elements


    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"matmul",output_shape,[],
                                        output_shape,[],ARITHMTIC_INTENSITY_TYPE.MEML)
    
    
    return calculate_block


def _activation_flops(input_shape: List, op_type: str,attributes: Dict) -> Tuple[float, Tuple[float, float, float]]:
    """
    激活函数: Relu, Sigmoid, Softmax, Erf, Not
    """
    shape = [1,1,1,1,1]
    for i in range(len(input_shape[0])-1, -1, -1):
        try:
            shape[i] = input_shape[0][i]
        except:
            oo = 0
    # try:
    #     N, C_in, H_in, W_in = input_shape[0]
    # except Exception as e:
    #     print("-"*20,input_shape[0])
    #     return 0
    _,N, C_in, H_in, W_in = shape
    num_elements = N * C_in * H_in * W_in
    
    # 不同激活函数的复杂度
    flops_factors = {
        'Relu': 1,       # 1 FLOP (比较)
        'Sigmoid': 4,    # exp, add, div (~4 FLOPs)
        'Softmax': 5,    # exp, sum, div (~5 FLOPs)
        'Erf': 10,       # 误差函数(复杂计算)
        'Not': 1,        # 逻辑非
    }
    
    # Softmax特殊处理: 需要计算整个通道
    if op_type == 'Softmax':
        # 默认沿着最后一个维度

        axis = attributes.get('axis', -1)
        dims = input_shape[0]
        axis_dim = dims[axis] if axis >= 0 else dims[axis + len(dims)]
        flops = num_elements * (axis_dim + 1)  # 指数 + 归一化
    else:
        flops = num_elements * flops_factors.get(op_type, 1)
    
    in_mem = num_elements
    w_mem = 0       # 矩阵乘法没有权重，都是特征
    out_mem = num_elements


    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"matmul",input_shape[0],[],
                                        input_shape[0],[],ARITHMTIC_INTENSITY_TYPE.MEML)
    
    
    return calculate_block

    

def _math_flops(input_shape: List, op_type: str,attributes: Dict) -> Tuple[float, Tuple[float, float, float]]:
    """
    数学函数: Sin, Cos, Sqrt
    """
    in_shape = input_shape[0]
    num_elements = 1
    for i in in_shape:
        num_elements *= i
        
    # 不同数学函数的复杂度
    flops_factors = {
        'Sin': 10,
        'Cos': 10,
        'Sqrt': 4,
    }
    
    flops = num_elements * flops_factors.get(op_type, 1)
    
    # 内存访问
    in_mem = num_elements
    w_mem = 0       # 矩阵乘法没有权重，都是特征
    out_mem = num_elements


    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"matmul",in_shape,[],
                                        in_shape,[],ARITHMTIC_INTENSITY_TYPE.SPEC)
    
    
    return calculate_block


def _shape_change_flops(input_shape: List, op_type: str, attributes: Dict) -> Tuple[float, Tuple[float, float, float]]:
    """
    形状变换算子: Reshape, Flatten, Squeeze, Unsqueeze, Transpose, Identity, Cast
    """

    in_shape = input_shape[0]
    num_elements = 1
    for i in in_shape:
        num_elements *= i
    
    # 特殊属性处理
    axis = attributes.get('axis', 0)
    
    # 计算输出尺寸
    if op_type == 'Reshape':
        # 第二个输入是形状
        output_shape = input_shape[1]
        # output_elements = np.prod(output_shape)
        output_elements = num_elements
    elif op_type == 'Flatten':
        output_elements = num_elements
    elif op_type == 'Squeeze':
        axes = attributes.get('axes', [])
        output_elements = num_elements  # 元素数不变
    elif op_type == 'Unsqueeze':
        axes = attributes.get('axes', [])
        output_elements = num_elements  # 元素数不变
    elif op_type == 'Transpose':
        # perm = attributes.get('perm', tuple(reversed(range(len(x.shape)))))
        output_elements = num_elements
    elif op_type in ['Identity', 'Cast']:
        output_elements = num_elements
    else:
        output_elements = num_elements
    
    # FLOPs (除类型转换外通常为0)
    flops = num_elements if op_type == 'Cast' else 0
    
    # 内存访问
    in_mem = num_elements
    w_mem = 0       # 矩阵乘法没有权重，都是特征
    out_mem = num_elements


    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"matmul",[],[],
                                        [],[],ARITHMTIC_INTENSITY_TYPE.MEML)
    
    
    return calculate_block

def _sequence_flops(input_shape: List, attributes: Dict, op_type: str) -> Tuple[float, Tuple[float, float, float]]:
    """
    序列操作: Concat, Split, Slice, Gather, Tile, Expand, Pad, CumSum
    """
    # 公共处理
    in_shape = input_shape[0]
    input_size = 1
    for i in in_shape:
        input_size *= i
    output_elements = input_size
    calculate_type = ARITHMTIC_INTENSITY_TYPE.MEML
    
    # 具体算子处理
    if op_type == 'Concat':
        # 沿指定轴连接多个张量
        axis = attributes.get('axis', 0)
        sizes = 0
        for inp in input_shape:
            size = 1
            for i in inp:
                size *= i
            sizes += size
        output_elements = sizes
    
    elif op_type == 'Split':
        # 沿轴切分张量
        axis = attributes.get('axis', 0)
        split = attributes.get('split', None)
        output_elements = input_size
    
    elif op_type == 'Slice':
        # 切片操作
        # starts = attributes['starts']
        # ends = attributes['ends']
        # axes = attributes.get('axes', list(range(len(starts))))
        # steps = attributes.get('steps', [1] * len(starts))
        
        # # 计算输出大小
        # output_shape = list(x.shape)
        # for i, axis in enumerate(axes):
        #     start, end, step = starts[i], ends[i], steps[i]
        #     size = (end - start) // step
        #     if size > 0:
        #         output_shape[axis] = size
        # output_elements = np.prod(output_shape)
        output_elements = input_size
    
    elif op_type == 'Gather':
        # 索引收集操作
        indices = input_shape[1]
        
        indices_size = 1
        for i in indices:
            indices_size *= i

        axis = attributes.get('axis', 0)
        output_elements = indices_size * (input_size // in_shape[axis])
        calculate_type = ARITHMTIC_INTENSITY_TYPE.IRMA
    
    elif op_type == 'Tile':
        # 瓦片复制
        # repeats = inputs[1]
        # output_shape = tuple(dim * repeat for dim, repeat in zip(x.shape, repeats))
        # output_elements = np.prod(output_shape)
        output_elements = input_size
    
    elif op_type == 'Expand':
        # 广播扩展
        # shape = inputs[1]
        # output_elements = np.prod(shape)
        output_elements = input_size
    
    elif op_type == 'Pad':
        # 填充操作
        # pads = inputs[1] if len(inputs) > 1 else attributes['pads']
        # output_shape = list(x.shape)
        # for i in range(len(x.shape)):
        #     output_shape[i] += pads[i] + pads[i + len(x.shape)]
        # output_elements = np.prod(output_shape)
        output_elements = input_size
    
    elif op_type == 'CumSum':
        # 累积和
        output_elements = input_size
    
    # FLOPs (通常每个元素一次操作)
    flops = output_elements
    
    # 内存访问
    in_mem = input_size
    w_mem = 0       # 矩阵乘法没有权重，都是特征
    out_mem = output_elements

    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"matmul",input_shape,[],
                                        input_shape,[],calculate_type)
    
    
    return calculate_block

def _reduction_flops(input_shape: List, attributes: Dict, op_type: str) -> Tuple[float, Tuple[float, float, float]]:
    """
    归约操作: MaxPool, GlobalAveragePool, ReduceMean, ReduceMax, Clip
    """
    shape = [1,1,1,1,1]
    for i in range(len(input_shape[0])-1, -1, -1):
        try:
            shape[i] = input_shape[0][i]
        except:
            oo = 0

    _,N, C_in, H_in, W_in = shape
    input_size = N * C_in * H_in * W_in
    output_elements = input_size  # 默认假设输出大小类似
    
    # 具体算子处理
    if op_type == 'MaxPool':
        # 最大池化
        kernel_shape = attributes['kernel_shape']
        strides = attributes.get('strides', [1, 1])
        pads = attributes.get('pads', [0, 0, 0, 0])
        
        N, C, H, W = N, C_in, H_in, W_in
        H_out = (H + pads[0] + pads[2] - kernel_shape[0]) // strides[0] + 1
        W_out = (W + pads[1] + pads[3] - kernel_shape[1]) // strides[1] + 1
        output_elements = N * C * H_out * W_out
    
    elif op_type == 'GlobalAveragePool':
        # 全局平均池化
        output_elements = N * C_in  # N x C
    
    elif op_type in ['ReduceMean', 'ReduceMax']:
        # 归约操作
        axes = attributes.get('axes', None)
        keepdims = attributes.get('keepdims', 1)
        
        if axes is None:
            output_elements = 1 if keepdims else input_size
        else:
            output_shape = shape
            for axis in axes:
                output_shape[axis] = 1
            output_elements = 1
            for i in output_shape:
                output_elements *= i
    
    elif op_type == 'Clip':
        # 裁剪操作
        min_val = input_shape[1] if len(input_shape) > 1 else attributes.get('min', -3.4028234663852886e+38)
        max_val = input_shape[2] if len(input_shape) > 2 else attributes.get('max', 3.4028234663852886e+38)
        output_elements = input_size
    
    # FLOPs
    if op_type == 'Clip':
        flops = output_elements * 2  # 两个比较操作
    elif op_type in ['MaxPool', 'ReduceMax']:
        flops = output_elements * np.prod(attributes.get('kernel_shape', [0]))  # 每个位置kernel_size次比较
    else:
        flops = output_elements  # 平均或均值池化每个位置一次操作
      
    # 内存访问
    in_mem = input_size
    w_mem = 0       # 矩阵乘法没有权重，都是特征
    out_mem = output_elements

    calculate_block = CalculateBlockInfo(in_mem,w_mem,out_mem,flops,"matmul",input_shape[0],[],
                                        input_shape[0],[],ARITHMTIC_INTENSITY_TYPE.REDM)
    
    
    return calculate_block

def _resize_flops(input_shape: List, attributes: Dict) -> Tuple[float, Tuple[float, float, float]]:
    """
    Resize操作
    """
    return None

    x = input_shape[0]
    input_size = 1
    
    # 根据模式确定复杂性
    mode = attributes.get('mode', 'nearest')
    
    # 计算输出尺寸
    if len(input_shape) > 3:
        output_shape = input_shape[3]
    else:
        # 简单假设尺寸加倍
        output_shape = tuple(int(dim * 2) for dim in input_shape[0])
    
    output_elements = 1
    for i in input_shape[0]:
        output_elements *= i

    
    # FLOPs（根据插值方法）
    flops_factors = {
        'nearest': 1,
        'linear': 2,
        'cubic': 4
    }
    flops = output_elements * flops_factors.get(mode, 1)
    
    # 内存访问
    feature_access = input_size
    weight_access = 0
    output_access = output_elements
    
    return flops, (feature_access, weight_access, output_access)

def _where_flops(input_shape: List) -> Tuple[float, Tuple[float, float, float]]:
    """
    Where条件选择操作
    """
    return None
    condition = inputs[0]
    x = inputs[1]
    y = inputs[2]
    
    # 获取广播后的输出尺寸
    shapes = [np.array(inp).shape for inp in inputs]
    max_dims = max(len(shape) for shape in shapes)
    padded_shapes = [list(shape) for shape in shapes]
    for shape in padded_shapes:
        while len(shape) < max_dims:
            shape.insert(0, 1)
    
    output_dims = []
    for dims in zip(*padded_shapes):
        output_dims.append(max(dims))
    output_shape = tuple(output_dims)
    output_elements = np.prod(output_shape)
    
    # FLOPs (每个元素一次条件判断+一个值加载)
    flops = output_elements
    
    # 内存访问
    feature_access = (np.array(condition).size + 
                      np.array(x).size + 
                      np.array(y).size)
    weight_access = 0
    output_access = output_elements
    
    return flops, (feature_access, weight_access, output_access)

#
# 
def _constant_flops(inputs: List, attributes: Dict) -> Tuple[float, Tuple[float, float, float]]:
    """
    常量操作: Constant, ConstantOfShape
    """
    return None
    if inputs:  # ConstantOfShape情况
        # 第一个输入是形状张量
        shape = inputs[0]
        output_elements = np.prod(shape)
    else:  # Constant情况
        value = attributes['value']
        output_elements = value.size
    
    # FLOPs (无)
    flops = 0
    
    # 内存访问
    feature_access = 0
    weight_access = output_elements  # 常量视为权重
    output_access = output_elements
    
    return flops, (feature_access, weight_access, output_access)

def _tensor_gen_flops(inputs: List, op_type: str) -> Tuple[float, Tuple[float, float, float]]:
    """
    张量生成: Range, Shape
    """
    return None
    if op_type == 'Range':
        # 输入: start, limit, delta
        start, limit, delta = inputs
        num_elements = (limit - start) // delta
        flops = 3  # 计算元素数
    else:  # Shape
        # 输入: data
        data = inputs[0]
        num_elements = len(data.shape)
        flops = 0
    
    # 内存访问
    feature_access = sum(np.array(inp).size for inp in inputs)
    weight_access = 0
    output_access = num_elements
    
    return flops, (feature_access, weight_access, output_access)

# --------------------------------------------------------------
# 测试函数
# --------------------------------------------------------------
def test_operator(op_type, inputs, attributes=None):
    """测试特定算子并打印结果"""
    try:
        flops, (feat_access, weight_access, output_access) = get_flops_and_memory_access(
            op_type, inputs, attributes
        )
        print(f"===== {op_type} Operator =====")
        print(f"Input Shapes: {[np.array(inp).shape for inp in inputs]}")
        print(f"FLOPs: {flops:.0f}")
        print(f"Memory Access (Features): {feat_access:.0f} elements")
        print(f"Memory Access (Weights): {weight_access:.0f} elements")
        print(f"Memory Access (Output): {output_access:.0f} elements")
        print("")
    except Exception as e:
        print(f"Error testing {op_type}: {str(e)}")

# if __name__ == "__main__":
#     # ===== 测试各个算子 =====
    
#     # Conv 测试
#     test_operator('Conv', [
#         np.ones((1, 3, 32, 32)),  # Input
#         np.ones((64, 3, 3, 3)),    # Weights
#         np.ones((64,))             # Bias
#     ], {'strides': [2, 2], 'pads': [1, 1, 1, 1]})
    
#     # MatMul 测试
#     test_operator('MatMul', [
#         np.ones((16, 256)),  # A
#         np.ones((256, 1024)) # B
#     ])
    
#     # Elementwise (Add) 测试
#     test_operator('Add', [
#         np.ones((1, 128, 28, 28)),  # Tensor1
#         np.ones((28, 28))            # Tensor2 (广播)
#     ])
    
#     # Activation (Sigmoid) 测试
#     test_operator('Sigmoid', [
#         np.ones((1, 128))
#     ])
    
#     # Reduction (GlobalAveragePool) 测试
#     test_operator('GlobalAveragePool', [
#         np.ones((1, 64, 32, 32))
#     ])
    
#     # Shape Change (Reshape) 测试
#     test_operator('Reshape', [
#         np.ones((1, 64, 32, 32)),  # Input
#         np.array([1, 64, 1024])    # New shape
#     ])
    
#     # Sequence (Slice) 测试
#     test_operator('Slice', [
#         np.ones((1, 3, 64, 64))
#     ], {'starts': [0, 0, 0, 0], 'ends': [1, 3, 32, 32]})
    
#     # Constant 测试
#     test_operator('Constant', [], {
#         'value': np.array([1, 2, 3])
#     })
    
#     # Where 测试
#     test_operator('Where', [
#         np.array([[True, False], [False, True]]),
#         np.array([[1, 2], [3, 4]]),
#         np.array([[10, 20], [30, 40]])
#     ])

