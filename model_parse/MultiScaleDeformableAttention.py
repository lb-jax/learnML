import math

def calculate_msda_complexity(
    batch_size: int,
    query_length: int,
    feature_levels: list, # List of (height, width) for each level, e.g., [(64,64), (32,32), (16,16), (8,8)]
    feature_dim: int,
    num_heads: int,
    num_sampling_points: int,
    dtype_bytes: int = 4 # float32 = 4 bytes
):
    """
    计算MultiScaleDeformableAttention前向传播的计算量 (FLOPs) 和访存量 (Bytes)。

    Args:
        batch_size (int): 批处理大小 B。
        query_length (int): 查询序列长度 Lq。
        feature_levels (list): 包含每个特征层 (height, width) 元组的列表，例如 [(64,64), (32,32), (16,16), (8,8)]。
        feature_dim (int): 特征维度 C。
        num_heads (int): 注意力头数 M。
        num_sampling_points (int): 每个注意力头在每个特征层采样的点数 P。
        dtype_bytes (int): 数据类型占用的字节数，默认为4 (float32)。

    Returns:
        tuple: (total_flops, total_memory_bytes)
    calculate_msda_complexity(1,19947, [(19947, 256), (19947, 256), (19947, 256), (19947, 256),(19947, 256)], 256, 8, 4)
    """

    B = batch_size
    Lq = query_length
    L_levels = len(feature_levels)
    C = feature_dim
    M = num_heads
    P = num_sampling_points
    C_head = C // M

    # 验证输入参数
    if C % M != 0:
        raise ValueError(f"Feature dimension (C={C}) must be divisible by num_heads (M={M}).")
    if L_levels == 0:
        raise ValueError("feature_levels cannot be empty.")

    # 计算总的value token数量
    total_value_tokens = sum([h * w for h, w in feature_levels])
    Lv = total_value_tokens # L_value_total

    # 存储总FLOPs和总访存量
    total_flops = 0
    total_memory_bytes = 0

    print(f"--- MultiScaleDeformableAttention Complexity Analysis ---")
    print(f"Parameters: B={B}, Lq={Lq}, C={C}, M={M}, P={P}, L_levels={L_levels}")
    print(f"C_head={C_head}, Total Value Tokens (Lv)={Lv}")
    print(f"Dtype Bytes: {dtype_bytes}")
    print("-" * 50)

    # 1. 线性投影 Query (用于生成偏移量和注意力权重)
    # 输入: query (B, Lq, C)
    # 输出: query_proj (B, Lq, C)
    # 权重: W_q (C, C), B_q (C,)
    step_name = "1. Linear Projection for Query"
    flops_step = B * Lq * (2 * C * C + C)
    memory_step = (B * Lq * C + # Read query
                   C * C + C + # Read W_q, B_q
                   B * Lq * C) * dtype_bytes # Write query_proj
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 矩阵乘法和偏置加法 (query @ W_q + B_q)")
    print(f"  - FLOPs: {flops_step:,}")
    print(f"  - 访存量: {memory_step:,} bytes (读 query, 读 W_q/B_q, 写 query_proj)")
    print("-" * 50)

    # 2. 线性投影 Value (用于提供采样特征)
    # 输入: value (B, Lv, C)
    # 输出: value_proj (B, Lv, C)
    # 权重: W_v (C, C), B_v (C,)
    step_name = "2. Linear Projection for Value"
    flops_step = B * Lv * (2 * C * C + C)
    memory_step = (B * Lv * C + # Read value
                   C * C + C + # Read W_v, B_v
                   B * Lv * C) * dtype_bytes # Write value_proj
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 矩阵乘法和偏置加法 (value @ W_v + B_v)")
    print(f"  - FLOPs: {flops_step:,}")
    print(f"  - 访存量: {memory_step:,} bytes (读 value, 读 W_v/B_v, 写 value_proj)")
    print("-" * 50)

    # 3. 预测采样点偏移量 (Sampling Offsets)
    # 输入: query_proj (B, Lq, C)
    # 输出: sampling_offsets (B, Lq, M, L_levels, P, 2) (2 for x,y coordinates)
    # 权重: W_offset (C, M * L_levels * P * 2), B_offset (M * L_levels * P * 2)
    output_dim_offsets = M * L_levels * P * 2
    step_name = "3. Predict Sampling Offsets"
    flops_step = B * Lq * (2 * C * output_dim_offsets + output_dim_offsets)
    memory_step = (B * Lq * C + # Read query_proj
                   C * output_dim_offsets + output_dim_offsets + # Read W_offset, B_offset
                   B * Lq * output_dim_offsets) * dtype_bytes # Write sampling_offsets
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 线性投影 (query_proj @ W_offset + B_offset)")
    print(f"  - FLOPs: {flops_step:,}")
    print(f"  - 访存量: {memory_step:,} bytes (读 query_proj, 读 W_offset/B_offset, 写 sampling_offsets)")
    print("-" * 50)

    # 4. 预测注意力权重 (Attention Weights)
    # 输入: query_proj (B, Lq, C)
    # 输出: attention_weights (B, Lq, M, L_levels, P)
    # 权重: W_attn (C, M * L_levels * P), B_attn (M * L_levels * P)
    output_dim_attn = M * L_levels * P
    step_name = "4. Predict Attention Weights"
    flops_linear = B * Lq * (2 * C * output_dim_attn + output_dim_attn)
    # Softmax over L_levels * P for each (B, Lq, M) group
    # Each softmax group has size L_levels * P. Approx 3 * N FLOPs for N elements.
    flops_softmax = B * Lq * M * (3 * L_levels * P)
    flops_step = flops_linear + flops_softmax
    memory_step = (B * Lq * C + # Read query_proj
                   C * output_dim_attn + output_dim_attn + # Read W_attn, B_attn
                   B * Lq * output_dim_attn) * dtype_bytes # Write attention_weights (intermediate before softmax)
    # Note: Softmax itself might involve intermediate writes, but we consider the final output size for minimum.
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 线性投影 (query_proj @ W_attn + B_attn) + Softmax")
    print(f"  - FLOPs (Linear): {flops_linear:,}")
    print(f"  - FLOPs (Softmax): {flops_softmax:,}")
    print(f"  - 总FLOPs: {flops_step:,}")
    print(f"  - 访存量: {memory_step:,} bytes (读 query_proj, 读 W_attn/B_attn, 写 attention_weights)")
    print("-" * 50)

    # 5. 计算实际采样位置 (Sampling Locations)
    # 输入: reference_points (B, Lq, 2), sampling_offsets (B, Lq, M, L_levels, P, 2), spatial_shapes (L_levels, 2)
    # 输出: sampling_locations (B, Lq, M, L_levels, P, 2)
    # 行为: (reference_points + sampling_offsets * level_scale)
    # reference_points需要扩展维度，offsets需要根据特征图大小进行归一化和缩放。
    # 每个采样点有两个坐标 (x, y)。
    num_coords = B * Lq * M * L_levels * P * 2
    step_name = "5. Calculate Sampling Locations"
    # 假设 scale_factor 是预计算好的，这里只计算加法和乘法
    flops_step = num_coords * 1 + num_coords * 1 # 1 for add, 1 for mul (scaling offset)
    memory_step = (B * Lq * 2 + # Read reference_points
                   B * Lq * M * L_levels * P * 2 + # Read sampling_offsets
                   num_coords) * dtype_bytes # Write sampling_locations
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 元素级加法和乘法 (参考点 + 偏移量 * 尺度因子)")
    print(f"  - FLOPs: {flops_step:,}")
    print(f"  - 访存量: {memory_step:,} bytes (读 reference_points, 读 sampling_offsets, 写 sampling_locations)")
    print("-" * 50)

    # 6. 特征采样 (Feature Sampling) - 通过grid_sample或类似操作
    # 输入: value_proj (B, Lv, C), sampling_locations (B, Lq, M, L_levels, P, 2)
    # 输出: sampled_values (B, Lq, M, L_levels, P, C_head)
    # 行为: 双线性插值。每个采样点的每个通道需要约 8 FLOPs (4点乘法+加法)。
    step_name = "6. Feature Sampling (Grid Sampling)"
    flops_step = B * Lq * M * L_levels * P * C_head * 8 # Approximation for bilinear interpolation
    memory_step = (B * Lv * C + # Read value_proj
                   B * Lq * M * L_levels * P * 2 + # Read sampling_locations
                   B * Lq * M * L_levels * P * C_head) * dtype_bytes # Write sampled_values
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 双线性插值 (从value_proj中根据采样位置提取特征)")
    print(f"  - FLOPs: {flops_step:,} (近似值)")
    print(f"  - 访存量: {memory_step:,} bytes (读 value_proj, 读 sampling_locations, 写 sampled_values)")
    print("-" * 50)

    # 7. 加权求和 (Weighted Sum)
    # 输入: sampled_values (B, Lq, M, L_levels, P, C_head), attention_weights (B, Lq, M, L_levels, P)
    # 输出: output_pooled (B, Lq, C) (经过sum over L_levels, P, 并拼接M头)
    # 行为: 元素级乘法 + 归约求和
    step_name = "7. Weighted Sum"
    # 元素级乘法 (sampled_values * attention_weights)
    flops_mul = B * Lq * M * L_levels * P * C_head
    # 求和 (sum over L_levels * P)
    # 每个 (B, Lq, M, C_head) 组需要对 L_levels * P 个元素求和
    flops_sum = B * Lq * M * C_head * (L_levels * P - 1)
    flops_step = flops_mul + flops_sum
    memory_step = (B * Lq * M * L_levels * P * C_head + # Read sampled_values
                   B * Lq * M * L_levels * P + # Read attention_weights
                   B * Lq * C) * dtype_bytes # Write output_pooled
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 元素级乘法 (权重 * 采样值) 和 归约求和")
    print(f"  - FLOPs (乘法): {flops_mul:,}")
    print(f"  - FLOPs (求和): {flops_sum:,}")
    print(f"  - 总FLOPs: {flops_step:,}")
    print(f"  - 访存量: {memory_step:,} bytes (读 sampled_values, 读 attention_weights, 写 output_pooled)")
    print("-" * 50)

    # 8. 最终线性投影 (Output Linear Projection)
    # 输入: output_pooled (B, Lq, C)
    # 输出: output (B, Lq, C)
    # 权重: W_out (C, C), B_out (C,)
    step_name = "8. Final Linear Projection"
    flops_step = B * Lq * (2 * C * C + C)
    memory_step = (B * Lq * C + # Read output_pooled
                   C * C + C + # Read W_out, B_out
                   B * Lq * C) * dtype_bytes # Write output
    total_flops += flops_step
    total_memory_bytes += memory_step
    print(f"{step_name}:")
    print(f"  - 计算行为: 矩阵乘法和偏置加法 (output_pooled @ W_out + B_out)")
    print(f"  - FLOPs: {flops_step:,}")
    print(f"  - 访存量: {memory_step:,} bytes (读 output_pooled, 读 W_out/B_out, 写 output)")
    print("-" * 50)

    print(f"\n--- Total Complexity ---")
    print(f"总计算量 (Total FLOPs): {total_flops:,}")
    print(f"总访存量 (Total Memory Access): {total_memory_bytes:,} bytes ({total_memory_bytes / (1024**3):.3f} GB)")

    return total_flops, total_memory_bytes

# --- 示例用法 ---
if __name__ == "__main__":
    # 典型参数 (以Deformable DETR为例，简化)
    # Query length: 100 for object queries + 1 for encoder output (optional)
    # Feature levels: usually P2, P3, P4, P5 from backbone (e.g., ResNet)
    # Feature dimension: 25**Considering Complexity Calculation**
    calculate_msda_complexity(1,19947, [(19947, 256), (19947, 256), (19947, 256), (19947, 256),(19947, 256)], 256, 8, 4)

