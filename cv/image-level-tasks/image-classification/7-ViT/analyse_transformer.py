import numpy as np

# 常量定义
FLOAT_BYTES = 4  # 32位浮点数占用字节数

def linear_flops_mem(in_features, out_features, input_shape, bias=True):
    """
    计算线性层(全连接层)的FLOPs和访存量
    
    参数:
        in_features: 输入特征数
        out_features: 输出特征数
        input_shape: 输入形状 (batch_size, seq_len, in_features)
        bias: 是否使用偏置
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状
    """
    batch_size, seq_len, _ = input_shape
    # 矩阵乘法计算量: (batch_size * seq_len) * in_features * out_features * 2 (乘加操作)
    flops = batch_size * seq_len * in_features * out_features * 2
    
    # 偏置加法
    if bias:
        flops += batch_size * seq_len * out_features * 2
    
    # 输出形状
    output_shape = (batch_size, seq_len, out_features)
    
    # 理论最低访存量: 输入 + 输出 + 权重 + 偏置
    input_mem = batch_size * seq_len * in_features * FLOAT_BYTES
    output_mem = batch_size * seq_len * out_features * FLOAT_BYTES
    weight_mem = in_features * out_features * FLOAT_BYTES
    bias_mem = out_features * FLOAT_BYTES if bias else 0
    
    mem_access = input_mem + output_mem + weight_mem + bias_mem
    
    return flops, mem_access, output_shape

def layer_norm_flops_mem(input_shape, elementwise_affine=True):
    """
    计算层归一化(包括Add & Norm操作)的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, seq_len, dim)
        elementwise_affine: 是否使用可学习的缩放和平移参数
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (与输入相同)
    """
    batch_size, seq_len, dim = input_shape
    output_shape = input_shape
    
    # 计算均值和方差: 每个元素计算2次 (计算均值、方差)
    # 每个特征维度都需要计算，所以是dim*2
    flops_per_token = dim * 2
    # 归一化操作: (x - mean) / sqrt(var + eps) * gamma + beta
    # 每个元素约5次操作 (减、除、乘、加、乘)
    flops_per_token += dim * 5
    
    # 总的FLOPs
    flops = batch_size * seq_len * flops_per_token
    
    # 访存量:
    # 输入: batch_size * seq_len * dim * 4
    # 输出: batch_size * seq_len * dim * 4
    # 可学习的参数: gamma和beta各dim
    param_mem = 0
    if elementwise_affine:
        param_mem = dim * 2 * FLOAT_BYTES
    
    mem_access = (batch_size * seq_len * dim * 2 * FLOAT_BYTES) + param_mem
    
    return flops, mem_access, output_shape

def softmax_flops_mem(input_shape, attention_mask=None):
    """
    计算softmax的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, num_heads, seq_len, seq_len)
        attention_mask: 注意力掩码 (如果有)
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (与输入相同)
    """
    batch_size, num_heads, q_seq_len, kv_seq_len = input_shape
    output_shape = input_shape
    
    # softmax计算量: 
    # 1. 计算指数: num_operations_per_element (约5次操作: max, exp, sum, div等)
    # 2. 每个元素都需要针对每行计算
    flops_per_row = kv_seq_len * 5  # 每行的softmax
    flops_per_token = flops_per_row * q_seq_len  # 每个查询位置的softmax
    
    # 总FLOPs
    flops = batch_size * num_heads * flops_per_token
    
    # 访存量: 输入 + 输出 + 临时存储 (指数值)
    input_mem = batch_size * num_heads * q_seq_len * kv_seq_len * FLOAT_BYTES
    output_mem = input_mem  # 输出与输入形状相同
    exp_mem = input_mem  # 临时存储指数值
    sum_mem = batch_size * num_heads * q_seq_len * FLOAT_BYTES  # 存储行和
    
    mem_access = input_mem + output_mem + exp_mem + sum_mem
    
    return flops, mem_access, output_shape

def self_attention_flops_mem(input_shape, d_model, num_heads, use_mask=False, is_causal=False):
    """
    计算自注意力机制的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, seq_len, d_model)
        d_model: 模型维度
        num_heads: 注意力头数
        use_mask: 是否使用注意力掩码
        is_causal: 是否是因果掩码 (仅解码器自注意力)
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, _ = input_shape
    d_k = d_model // num_heads  # 每个头的维度
    
    # 计算Q、K、V投影
    # 每个投影都是线性层，输入输出维度相同
    flops_q, mem_q, _ = linear_flops_mem(d_model, d_model, input_shape)
    flops_k, mem_k, _ = linear_flops_mem(d_model, d_model, input_shape)
    flops_v, mem_v, _ = linear_flops_mem(d_model, d_model, input_shape)
    
    # 计算QK^T: (batch_size, num_heads, seq_len, d_k) * (batch_size, num_heads, d_k, seq_len)
    # 每个矩阵乘: num_heads * seq_len * d_k * seq_len * 2
    flops_qk = batch_size * num_heads * seq_len * d_k * seq_len * 2
    
    # 缩放和掩码
    # 缩放只是一个标量乘法: batch_size * num_heads * seq_len * seq_len
    flops_scale = batch_size * num_heads * seq_len * seq_len * 1
    
    # 计算softmax
    softmax_shape = (batch_size, num_heads, seq_len, seq_len)
    flops_softmax, mem_softmax, _ = softmax_flops_mem(softmax_shape)
    
    # 计算注意力加权值: attn * V
    # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, d_k)
    flops_av = batch_size * num_heads * seq_len * seq_len * d_k * 2
    
    # 最终输出投影
    flops_output, mem_output, output_shape = linear_flops_mem(d_model, d_model, 
                                                             (batch_size, seq_len, d_model))
    
    # 总FLOPs
    total_flops = (flops_q + flops_k + flops_v + flops_qk + flops_scale + 
                   flops_softmax + flops_av + flops_output)
    
    # 如果有掩码，增加额外的计算量
    if use_mask or is_causal:
        # 掩码操作的计算量相对较小
        mask_flops = batch_size * num_heads * seq_len * seq_len * 1
        total_flops += mask_flops
    
    # 访存量 (简化计算)
    # 主要考虑权重和中间结果
    # 投影层权重: 4 * (d_model * d_model) * FLOAT_BYTES (Q, K, V, O)
    # 投影层中间结果: Q, K, V, attn_logits, attn_weights, attn_output, output
    qkv_mem = batch_size * seq_len * d_model * FLOAT_BYTES * 3  # Q, K, V
    attn_logits_mem = batch_size * num_heads * seq_len * seq_len * FLOAT_BYTES
    attn_weights_mem = attn_logits_mem
    attn_output_mem = batch_size * num_heads * seq_len * d_k * FLOAT_BYTES
    
    # 总访存量
    total_mem = (mem_q + mem_k + mem_v + mem_output + 
                 qkv_mem + attn_logits_mem + attn_weights_mem + attn_output_mem)
    
    return total_flops, total_mem, output_shape

def positionwise_ffn_flops_mem(input_shape, d_model, d_ff):
    """
    计算位置级前馈网络(FFN)的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, seq_len, d_model)
        d_model: 模型维度
        d_ff: 前馈层内部维度
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (与输入相同)
    """
    batch_size, seq_len, _ = input_shape
    
    # 第一个线性层: 扩展维度
    flops1, mem1, _ = linear_flops_mem(d_model, d_ff, input_shape)
    
    # ReLU激活: 计算量很小，但访存重要
    flops_relu = batch_size * seq_len * d_ff  # 比较操作
    mem_relu = batch_size * seq_len * d_ff * FLOAT_BYTES * 2  # 输入和输出
    
    # 第二个线性层: 降维回d_model
    flops2, mem2, output_shape = linear_flops_mem(d_ff, d_model, 
                                                (batch_size, seq_len, d_ff))
    
    # 总FLOPs
    total_flops = flops1 + flops_relu + flops2
    
    # 总访存量
    total_mem = mem1 + mem_relu + mem2
    
    return total_flops, total_mem, output_shape

def encoder_layer_flops_mem(input_shape, d_model, num_heads, d_ff):
    """
    计算单个Encoder层的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, seq_len, d_model)
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈层内部维度
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (与输入相同)
    """
    batch_size, seq_len, _ = input_shape
    
    # 自注意力子层
    flops_attn, mem_attn, _ = self_attention_flops_mem(input_shape, d_model, num_heads)
    
    # Add & Norm1: 残差连接 + 层归一化
    # 残差连接的访存: 读两个特征图 + 写输出
    residual_mem1 = batch_size * seq_len * d_model * FLOAT_BYTES * 3
    flops_norm1, mem_norm1, _ = layer_norm_flops_mem(input_shape)
    
    # 前馈网络
    flops_ffn, mem_ffn, _ = positionwise_ffn_flops_mem(input_shape, d_model, d_ff)
    
    # Add & Norm2: 残差连接 + 层归一化
    residual_mem2 = batch_size * seq_len * d_model * FLOAT_BYTES * 3
    flops_norm2, mem_norm2, _ = layer_norm_flops_mem(input_shape)
    
    # 总FLOPs
    total_flops = flops_attn + flops_norm1 + flops_ffn + flops_norm2
    
    # 总访存量
    total_mem = (mem_attn + residual_mem1 + mem_norm1 + 
                mem_ffn + residual_mem2 + mem_norm2)
    
    return total_flops, total_mem, input_shape

def decoder_self_attention_flops_mem(input_shape, d_model, num_heads):
    """
    计算解码器的自注意力层(因果掩码)的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, seq_len, d_model)
        d_model: 模型维度
        num_heads: 注意力头数
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (batch_size, seq_len, d_model)
    """
    # 由于因果掩码需要，计算与常规自注意力略有不同
    batch_size, seq_len, _ = input_shape
    d_k = d_model // num_heads
    
    # Q、K、V投影
    flops_q, mem_q, _ = linear_flops_mem(d_model, d_model, input_shape)
    flops_k, mem_k, _ = linear_flops_mem(d_model, d_model, input_shape)
    flops_v, mem_v, _ = linear_flops_mem(d_model, d_model, input_shape)
    
    # 因果QK^T: 只能访问到左侧位置
    # 有效计算量约为常规自注意力的一半
    flops_qk = batch_size * num_heads * seq_len * d_k * (seq_len // 2) * 2
    
    # 缩放
    flops_scale = batch_size * num_heads * seq_len * (seq_len // 2)
    
    # Softmax: 也相应减少
    softmax_shape = (batch_size, num_heads, seq_len, seq_len // 2)
    flops_softmax, mem_softmax, _ = softmax_flops_mem(softmax_shape)
    
    # 注意力加权值
    flops_av = batch_size * num_heads * seq_len * (seq_len // 2) * d_k * 2
    
    # 输出投影
    flops_output, mem_output, output_shape = linear_flops_mem(d_model, d_model, 
                                                             (batch_size, seq_len, d_model))
    
    # 总FLOPs
    total_flops = (flops_q + flops_k + flops_v + flops_qk + flops_scale + 
                   flops_softmax + flops_av + flops_output)
    
    # 访存量 (简化)
    total_mem = mem_q + mem_k + mem_v + mem_output + mem_softmax
    
    return total_flops, total_mem, output_shape

def encoder_decoder_attention_flops_mem(dec_input_shape, enc_output_shape, d_model, num_heads):
    """
    计算编码器-解码器注意力层的FLOPs和访存量
    
    参数:
        dec_input_shape: 解码器输入形状 (batch_size, tgt_seq_len, d_model)
        enc_output_shape: 编码器输出形状 (batch_size, src_seq_len, d_model)
        d_model: 模型维度
        num_heads: 注意力头数
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (batch_size, tgt_seq_len, d_model)
    """
    batch_size, tgt_seq_len, _ = dec_input_shape
    _, src_seq_len, _ = enc_output_shape
    d_k = d_model // num_heads
    
    # Q投影 (来自解码器输入)
    flops_q, mem_q, _ = linear_flops_mem(d_model, d_model, dec_input_shape)
    
    # K、V投影 (来自编码器输出)
    flops_k, mem_k, _ = linear_flops_mem(d_model, d_model, enc_output_shape)
    flops_v, mem_v, _ = linear_flops_mem(d_model, d_model, enc_output_shape)
    
    # QK^T计算: (batch_size, num_heads, tgt_seq_len, d_k) * (batch_size, num_heads, d_k, src_seq_len)
    flops_qk = batch_size * num_heads * tgt_seq_len * d_k * src_seq_len * 2
    
    # 缩放
    flops_scale = batch_size * num_heads * tgt_seq_len * src_seq_len
    
    # Softmax
    softmax_shape = (batch_size, num_heads, tgt_seq_len, src_seq_len)
    flops_softmax, mem_softmax, _ = softmax_flops_mem(softmax_shape)
    
    # 注意力加权值
    flops_av = batch_size * num_heads * tgt_seq_len * src_seq_len * d_k * 2
    
    # 输出投影
    flops_output, mem_output, output_shape = linear_flops_mem(d_model, d_model, 
                                                            (batch_size, tgt_seq_len, d_model))
    
    # 总FLOPs
    total_flops = (flops_q + flops_k + flops_v + flops_qk + flops_scale + 
                   flops_softmax + flops_av + flops_output)
    
    # 访存量
    total_mem = mem_q + mem_k + mem_v + mem_output + mem_softmax
    
    return total_flops, total_mem, output_shape

def decoder_layer_flops_mem(dec_input_shape, enc_output_shape, d_model, num_heads, d_ff):
    """
    计算单个Decoder层的FLOPs和访存量
    
    参数:
        dec_input_shape: 解码器输入形状 (batch_size, seq_len, d_model)
        enc_output_shape: 编码器输出形状 (batch_size, src_seq_len, d_model)
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈层内部维度
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, _ = dec_input_shape
    
    # 解码器自注意力 (带因果掩码)
    flops_self_attn, mem_self_attn, _ = decoder_self_attention_flops_mem(
        dec_input_shape, d_model, num_heads)
    
    # Add & Norm1
    residual_mem1 = batch_size * seq_len * d_model * FLOAT_BYTES * 3
    flops_norm1, mem_norm1, _ = layer_norm_flops_mem(dec_input_shape)
    
    # 编码器-解码器注意力
    flops_encdec_attn, mem_encdec_attn, _ = encoder_decoder_attention_flops_mem(
        dec_input_shape, enc_output_shape, d_model, num_heads)
    
    # Add & Norm2
    residual_mem2 = batch_size * seq_len * d_model * FLOAT_BYTES * 3
    flops_norm2, mem_norm2, _ = layer_norm_flops_mem(dec_input_shape)
    
    # 前馈网络
    flops_ffn, mem_ffn, _ = positionwise_ffn_flops_mem(dec_input_shape, d_model, d_ff)
    
    # Add & Norm3
    residual_mem3 = batch_size * seq_len * d_model * FLOAT_BYTES * 3
    flops_norm3, mem_norm3, _ = layer_norm_flops_mem(dec_input_shape)
    
    # 总FLOPs
    total_flops = (flops_self_attn + flops_norm1 + 
                  flops_encdec_attn + flops_norm2 + 
                  flops_ffn + flops_norm3)
    
    # 总访存量
    total_mem = (mem_self_attn + residual_mem1 + mem_norm1 + 
                mem_encdec_attn + residual_mem2 + mem_norm2 + 
                mem_ffn + residual_mem3 + mem_norm3)
    
    return total_flops, total_mem, dec_input_shape

def embedding_flops_mem(vocab_size, d_model, batch_size, seq_len):
    """
    计算嵌入层的FLOPs和访存量
    
    参数:
        vocab_size: 词汇表大小
        d_model: 模型维度
        batch_size: 批大小
        seq_len: 序列长度
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (batch_size, seq_len, d_model)
    """
    # 嵌入层主要是一个查表操作，计算量很小
    flops = 0  # 查表操作通常不计算为FLOPs
    
    # 访存量: 输入序列 + 嵌入矩阵 + 输出嵌入
    # 输入序列: 通常是整数类型，每个token用4字节
    input_mem = batch_size * seq_len * FLOAT_BYTES
    
    # 嵌入矩阵: 通常只访问实际使用的行
    # 实际访问嵌入向量数 = batch_size * seq_len
    weight_access = batch_size * seq_len * d_model * FLOAT_BYTES
    
    # 输出嵌入: batch_size * seq_len * d_model * FLOAT_BYTES
    output_mem = batch_size * seq_len * d_model * FLOAT_BYTES
    
    mem_access = input_mem + weight_access + output_mem
    output_shape = (batch_size, seq_len, d_model)
    
    return flops, mem_access, output_shape

def positional_encoding_flops_mem(input_shape):
    """
    计算位置编码的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, seq_len, d_model)
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (与输入相同)
    """
    batch_size, seq_len, d_model = input_shape
    
    # 位置编码主要是在嵌入向量上加上位置信息
    # 实际计算量很小，可以忽略
    flops = 0
    
    # 访存量: 输入 + 输出 + 位置编码矩阵
    input_mem = batch_size * seq_len * d_model * FLOAT_BYTES
    output_mem = input_mem  # 输出大小与输入相同
    pe_mem = seq_len * d_model * FLOAT_BYTES  # 位置编码矩阵
    
    mem_access = input_mem + output_mem + pe_mem
    output_shape = input_shape
    
    return flops, mem_access, output_shape

def output_layer_flops_mem(input_shape, vocab_size):
    """
    计算输出层(线性层+softmax)的FLOPs和访存量
    
    参数:
        input_shape: 输入形状 (batch_size, seq_len, d_model)
        vocab_size: 词汇表大小
        
    返回:
        flops: 浮点运算次数
        mem_access: 理论最低访存量 (bytes)
        output_shape: 输出形状 (batch_size, seq_len, vocab_size)
    """
    batch_size, seq_len, d_model = input_shape
    output_shape = (batch_size, seq_len, vocab_size)
    
    # 线性层
    flops_linear, mem_linear, logits_shape = linear_flops_mem(
        d_model, vocab_size, input_shape
    )
    
    # Softmax: 沿词汇表维度
    # 计算量: 每个token的softmax大约5*vocab_size次操作
    flops_softmax = batch_size * seq_len * vocab_size * 5
    
    # 访存量: logits + 输出概率
    logits_mem = batch_size * seq_len * vocab_size * FLOAT_BYTES
    output_mem = logits_mem
    
    # 总FLOPs
    total_flops = flops_linear + flops_softmax
    
    # 总访存量
    total_mem = mem_linear + logits_mem + output_mem
    
    return total_flops, total_mem, output_shape

def transformer_flops_mem(
    src_vocab_size, 
    tgt_vocab_size, 
    src_seq_len, 
    tgt_seq_len, 
    batch_size=1,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    use_positional_encoding=True,
    output_layer=True
):
    """
    计算整个Transformer模型的FLOPs和访存量
    
    参数:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        src_seq_len: 源序列长度
        tgt_seq_len: 目标序列长度
        batch_size: 批大小
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈层内部维度
        num_encoder_layers: Encoder层数
        num_decoder_layers: Decoder层数
        use_positional_encoding: 是否使用位置编码
        output_layer: 是否包含输出层
        
    返回:
        total_flops: 总浮点运算次数
        total_mem: 总理论访存量 (bytes)
        part_results: 各部件计算量和访存量明细
    """
    # 用于存储各部分结果
    part_results = {
        'encoder': {'embedding': 0, 'positional': 0, 'layers': 0, 'total_flops': 0, 'total_mem': 0},
        'decoder': {'embedding': 0, 'positional': 0, 'layers': 0, 'total_flops': 0, 'total_mem': 0},
        'output_layer': 0,
        'total_flops': 0,
        'total_mem': 0
    }
    
    # 存储中间形状
    shapes = {}
    
    ########################
    # Encoder部分
    ########################
    # 输入嵌入层
    flops, mem, shape = embedding_flops_mem(src_vocab_size, d_model, batch_size, src_seq_len)
    shapes['encoder_embed'] = shape
    part_results['encoder']['embedding'] = {'flops': flops, 'mem': mem}
    
    # 位置编码
    if use_positional_encoding:
        flops, mem, shape = positional_encoding_flops_mem(shape)
        shapes['encoder_pe'] = shape
        part_results['encoder']['positional'] = {'flops': flops, 'mem': mem}
    else:
        shapes['encoder_pe'] = shapes['encoder_embed']
    
    # Encoder层
    encoder_layers_flops = 0
    encoder_layers_mem = 0
    
    encoder_output_shape = shapes['encoder_pe']
    for _ in range(num_encoder_layers):
        flops, mem, encoder_output_shape = encoder_layer_flops_mem(
            encoder_output_shape, d_model, num_heads, d_ff
        )
        encoder_layers_flops += flops
        encoder_layers_mem += mem
    
    shapes['encoder_output'] = encoder_output_shape
    part_results['encoder']['layers'] = {'flops': encoder_layers_flops, 'mem': encoder_layers_mem}
    
    # Encoder总计算量和访存量
    encoder_total_flops = (
        part_results['encoder']['embedding']['flops'] + 
        (part_results['encoder']['positional']['flops'] if use_positional_encoding else 0) + 
        encoder_layers_flops
    )
    
    encoder_total_mem = (
        part_results['encoder']['embedding']['mem'] + 
        (part_results['encoder']['positional']['mem'] if use_positional_encoding else 0) + 
        encoder_layers_mem
    )
    
    part_results['encoder']['total_flops'] = encoder_total_flops
    part_results['encoder']['total_mem'] = encoder_total_mem
    
    ########################
    # Decoder部分
    ########################
    # 目标嵌入层
    flops, mem, shape = embedding_flops_mem(tgt_vocab_size, d_model, batch_size, tgt_seq_len)
    shapes['decoder_embed'] = shape
    part_results['decoder']['embedding'] = {'flops': flops, 'mem': mem}
    
    # 目标位置编码
    if use_positional_encoding:
        flops, mem, shape = positional_encoding_flops_mem(shape)
        shapes['decoder_pe'] = shape
        part_results['decoder']['positional'] = {'flops': flops, 'mem': mem}
    else:
        shapes['decoder_pe'] = shapes['decoder_embed']
    
    # Decoder层
    decoder_layers_flops = 0
    decoder_layers_mem = 0
    
    decoder_output_shape = shapes['decoder_pe']
    for _ in range(num_decoder_layers):
        flops, mem, decoder_output_shape = decoder_layer_flops_mem(
            decoder_output_shape, 
            shapes['encoder_output'],  # encoder输出作为decoder的输入
            d_model, num_heads, d_ff
        )
        decoder_layers_flops += flops
        decoder_layers_mem += mem
    
    shapes['decoder_output'] = decoder_output_shape
    part_results['decoder']['layers'] = {'flops': decoder_layers_flops, 'mem': decoder_layers_mem}
    
    # Decoder总计算量和访存量
    decoder_total_flops = (
        part_results['decoder']['embedding']['flops'] + 
        (part_results['decoder']['positional']['flops'] if use_positional_encoding else 0) + 
        decoder_layers_flops
    )
    
    decoder_total_mem = (
        part_results['decoder']['embedding']['mem'] + 
        (part_results['decoder']['positional']['mem'] if use_positional_encoding else 0) + 
        decoder_layers_mem
    )
    
    part_results['decoder']['total_flops'] = decoder_total_flops
    part_results['decoder']['total_mem'] = decoder_total_mem
    
    ########################
    # 输出层
    ########################
    if output_layer:
        flops, mem, _ = output_layer_flops_mem(shapes['decoder_output'], tgt_vocab_size)
        part_results['output_layer'] = {'flops': flops, 'mem': mem}
    else:
        part_results['output_layer'] = {'flops': 0, 'mem': 0}
    
    ########################
    # 汇总结果
    ########################
    total_flops = (
        encoder_total_flops + 
        decoder_total_flops + 
        part_results['output_layer']['flops']
    )
    
    total_mem = (
        encoder_total_mem + 
        decoder_total_mem + 
        part_results['output_layer']['mem']
    )
    
    part_results['total_flops'] = total_flops
    part_results['total_mem'] = total_mem
    
    return total_flops, total_mem, part_results

# 主函数: 计算原生Transformer的参数并输出结果
if __name__ == "__main__":
    # 原生Transformer的参数配置 ("Attention is All You Need")
    config = {
        'src_vocab_size': 37000,  # 英语-德语翻译常用词汇表大小
        'tgt_vocab_size': 37000,
        'src_seq_len': 192,       # 典型序列长度
        'tgt_seq_len': 100,
        'batch_size': 1,          # 单次推理
        'd_model': 768,
        'num_heads': 12,
        'd_ff': 3072,
        'num_encoder_layers': 12,
        'num_decoder_layers': 0,
        'use_positional_encoding': True,
        'output_layer': True
    }
    
    total_flops, total_mem, parts = transformer_flops_mem(**config)
    
    # # 打印结果
    # print(f"Transformer FLOPs and Memory Access Breakdown:")
    # print(f"Total FLOPs: {total_flops / 1e9:.2f} G")
    # print(f"Total Memory Access: {total_mem / (1024**2):.2f} MB")
    
    # 格式化函数
    def format_flops(flops):
        return f"{flops / 1e9:.4f} G" if flops > 1e9 else f"{flops / 1e6:.2f} M"
    
    def format_mem(mem):
        return f"{mem / (1024**2):.2f} MB"
    
    # # 打印详细分解
    # print("\nEncoder Breakdown:")
    # print(f"  Embedding: {format_flops(parts['encoder']['embedding']['flops'])} | {format_mem(parts['encoder']['embedding']['mem'])}")
    # print(f"  Positional: {format_flops(parts['encoder']['positional']['flops'])} | {format_mem(parts['encoder']['positional']['mem'])}")
    # print(f"  Layers: {format_flops(parts['encoder']['layers']['flops'])} | {format_mem(parts['encoder']['layers']['mem'])}")
    # print(f"Encoder Total: {format_flops(parts['encoder']['total_flops'])} | {format_mem(parts['encoder']['total_mem'])}")
    
    # print("\nDecoder Breakdown:")
    # print(f"  Embedding: {format_flops(parts['decoder']['embedding']['flops'])} | {format_mem(parts['decoder']['embedding']['mem'])}")
    # print(f"  Positional: {format_flops(parts['decoder']['positional']['flops'])} | {format_mem(parts['decoder']['positional']['mem'])}")
    # print(f"  Layers: {format_flops(parts['decoder']['layers']['flops'])} | {format_mem(parts['decoder']['layers']['mem'])}")
    # print(f"Decoder Total: {format_flops(parts['decoder']['total_flops'])} | {format_mem(parts['decoder']['total_mem'])}")
    
    # print(f"\nOutput Layer: {format_flops(parts['output_layer']['flops'])} | {format_mem(parts['output_layer']['mem'])}")
    # print(f"\nOverall: {format_flops(total_flops)} | {format_mem(total_mem)}")

    # print("-" * 50)
    print(f"{'Total':<15} | {format_flops(total_flops):>15} | {format_mem(total_mem):>15}")
    
    # 计算计算访存比
    flops_mem_ratio = total_flops / (total_mem /4)  # FLOPs per byte
    print(f"\nCompute-Memory Ratio (FLOPs/byte): {flops_mem_ratio:.4f}")

    