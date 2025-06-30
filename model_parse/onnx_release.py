import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, shape_inference, numpy_helper, TensorProto
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx-tracer")

def get_operator_details(
    model_path: str, 
    input_data: Dict[str, np.ndarray],
    skip_unimplemented: bool = True,
    customize_attributes: bool = True
) -> List[Dict[str, Any]]:
    """
    获取ONNX模型推理过程中所有算子的详细计算信息
    
    参数:
        model_path: ONNX模型文件路径
        input_data: 模型输入数据(字典: {输入名称: numpy数组})
        skip_unimplemented: 是否跳过不支持的操作符
        customize_attributes: 是否自定义属性解析
        
    返回:
        包含每个算子详细信息的字典列表
    """
    # 1. 加载模型并进行形状推理
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    
    # 2. 准备运行时数据采集模型
    trace_model = prepare_tracing_model(model)
    
    # 3. 创建推理会话
    session = create_inference_session(trace_model)
    
    # 4. 准备输入数据
    verified_inputs = {}
    for inp in session.get_inputs():
        if inp.name in input_data:
            verified_inputs[inp.name] = verify_input_data(
                input_data[inp.name], 
                tuple(inp.shape) if inp.shape else None, 
                inp.type
            )
        else:
            logger.warning(f"缺少输入: {inp.name}，尝试使用零张量填充")
            verified_inputs[inp.name] = create_default_input(inp.shape, inp.type)
    
    # 5. 运行推理并收集所有输出
    all_outputs = session.run(None, verified_inputs)
    output_names = [out.name for out in session.get_outputs()]
    
    # 6. 构建完整张量数据映射
    tensor_data = dict(zip(output_names, all_outputs))
    tensor_data.update(verified_inputs)
    
    # 7. 收集所有初始值
    for init in model.graph.initializer:
        tensor_data[init.name] = numpy_helper.to_array(init)
    
    # 8. 提取算子详细信息
    operator_details = []
    for node in model.graph.node:
        try:
            node_info = process_node(node, tensor_data, customize_attributes)
            operator_details.append(node_info)
        except Exception as e:
            if skip_unimplemented:
                logger.warning(f"跳过节点 {node.name} ({node.op_type}): {str(e)}")
            else:
                raise
                
    return operator_details

def prepare_tracing_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    创建用于追踪的模型，确保所有中间输出都包含
    并且有正确的数据类型
    """
    # 创建一个新模型副本
    trace_model = onnx.ModelProto()
    trace_model.CopyFrom(model)
    
    # 使用原始模型的输出节点名称
    original_outputs = {out.name for out in model.graph.output}
    
    # 添加所有中间节点作为输出
    value_info_map = build_value_info_map(model)
    for node in model.graph.node:
        for output in node.output:
            if output in original_outputs:
                continue  # 已经是最终输出
                
            # 获取节点输出的类型信息
            if output in value_info_map:
                value_info = value_info_map[output]
            else:
                # 为未知节点创建合理的默认值
                logger.warning(f"无法确定输出节点 '{output}' 的类型信息，使用默认值")
                value_info = helper.make_tensor_value_info(
                    output, TensorProto.FLOAT, None  # shape unknown
                )
                
            # 添加到追踪模型
            if not any(out.name == output for out in trace_model.graph.output):
                trace_model.graph.output.extend([value_info])
                
    # 清理可能的重复输出
    trace_model = remove_duplicate_outputs(trace_model)
    
    return trace_model

def build_value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    """构建张量名称到ValueInfo的映射"""
    value_info_map = {}
    
    # 添加显式值信息
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi
        
    # 添加输入
    for inp in model.graph.input:
        value_info_map[inp.name] = inp
        
    # 添加输出
    for out in model.graph.output:
        value_info_map[out.name] = out
        
    return value_info_map

def remove_duplicate_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """删除重复的输出节点"""
    clean_model = onnx.ModelProto()
    clean_model.CopyFrom(model)
    clean_model.graph.ClearField("output")
    
    seen = set()
    for out in model.graph.output:
        if out.name not in seen:
            clean_model.graph.output.extend([out])
            seen.add(out.name)
            
    return clean_model

def verify_input_data(
    data: np.ndarray, 
    expected_shape: Optional[Tuple], 
    expected_type: str
) -> np.ndarray:
    """
    验证输入数据是否符合模型要求
    并执行必要的转换
    """
    # 检查数据类型
    if "float" in expected_type:
        if not np.issubdtype(data.dtype, np.floating):
            logger.warning(f"强制转换输入到float32 (预期 {expected_type})")
            return data.astype(np.float32)
    
    elif "int" in expected_type:
        if not np.issubdtype(data.dtype, np.integer):
            logger.warning(f"强制转换输入到int64 (预期 {expected_type})")
            return data.astype(np.int64)
    
    # 检查形状 (如果模型指定了形状)
    if expected_shape is not None:
        # 动态形状处理：用-1表示可变维度
        if any(dim == -1 for dim in expected_shape):
            return data
            
        if len(data.shape) != len(expected_shape):
            raise ValueError(f"形状维度不匹配: {data.shape} vs {expected_shape}")
            
        for i, (actual, expected) in enumerate(zip(data.shape, expected_shape)):
            try:
                if type(actual) != type(expected):
                    logger.warning(f"维度{i}类型不匹配: {actual} != {expected}")
                    
                elif expected > 0 and actual != expected:
                    logger.warning(f"维度{i}形状不匹配: {actual} != {expected}")
            except:
                logger.error(f"-----"*10)
                logger.error(f"actual:{actual}")
                logger.error(f"expected:{expected}")
    
    return data

def create_default_input(shape: Optional[Tuple], dtype: str) -> np.ndarray:
    """创建默认输入张量"""
    # 处理未知形状
    if shape is None:
        logger.warning("创建默认输入张量: 未知形状, 使用 [1]")
        shape = (1,)
    
    # 处理动态维度
    fixed_shape = []
    for dim in shape:
        if dim == -1 or dim is None:
            fixed_shape.append(1)  # 为未知维度使用1
        else:
            fixed_shape.append(dim)
    
    # 确定数据类型
    if "float" in dtype:
        return np.zeros(fixed_shape, dtype=np.float32)
    elif "int64" in dtype:
        return np.zeros(fixed_shape, dtype=np.int64)
    elif "int32" in dtype:
        return np.zeros(fixed_shape, dtype=np.int32)
    else:
        logger.warning(f"未知数据类型: {dtype}, 使用float32")
        return np.zeros(fixed_shape, dtype=np.float32)

def create_inference_session(model: onnx.ModelProto) -> ort.InferenceSession:
    """创建ONNX Runtime会话"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.log_severity_level = 3  # 错误级日志
    
    # 序列化模型用于ORT加载
    model_bytes = model.SerializeToString()
    
    # 尝试使用不同执行提供者
    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
        "TensorrtExecutionProvider"
    ]
    
    for provider in providers:
        try:
            session = ort.InferenceSession(model_bytes, sess_options, providers=[provider])
            logger.info(f"使用执行提供者: {provider}")
            return session
        except:
            logger.warning(f"提供者 {provider} 不可用，尝试下一个")
    
    # 如果所有提供者都失败，使用默认
    logger.warning("无法加载任何提供者，使用默认")
    return ort.InferenceSession(model_bytes, sess_options)

def process_node(
    node: onnx.NodeProto, 
    tensor_data: Dict[str, np.ndarray],
    customize_attributes: bool
) -> Dict[str, Any]:
    """处理单个节点，提取运行时信息"""
    # 收集输入数据
    inputs = {}
    for name in node.input:
        if name in tensor_data:
            inputs[name] = tensor_data[name]
        else:
            logger.warning(f"找不到输入张量: {name} (节点: {node.name})")
            inputs[name] = None
    
    # 收集输出数据
    outputs = {}
    for name in node.output:
        if name in tensor_data:
            outputs[name] = tensor_data[name]
        else:
            logger.warning(f"找不到输出张量: {name} (节点: {node.name})")
            outputs[name] = None
    
    # 收集属性
    attributes = {}
    if customize_attributes:
        for attr in node.attribute:
            attributes[attr.name] = parse_attribute(attr)
    
    return {
        "node": {
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain
        },
        "inputs": inputs,
        "outputs": outputs,
        "attributes": attributes
    }

def parse_attribute(attr: onnx.AttributeProto) -> Any:
    """解析ONNX属性值"""
    if attr.HasField('f'): return float(attr.f)
    if attr.HasField('i'): return int(attr.i)
    if attr.HasField('s'): return attr.s.decode('utf-8')
    if attr.HasField('t'): return numpy_helper.to_array(attr.t)
    if attr.HasField('g'): return attr.g
    if len(attr.floats): return list(map(float, attr.floats))
    if len(attr.ints): return list(map(int, attr.ints))
    if len(attr.strings): return [s.decode('utf-8') for s in attr.strings]
    return None


def plot_int_distribution(data, output_path="./out/int_distribution.png", title="Integer Value Distribution"):
    """
    将整数列表按数值分类，绘制柱状图并保存（数值不连续时柱子紧挨显示）
    
    参数:
    data -- 包含整数的列表
    output_path -- 图像保存路径 (默认为 'int_distribution.png')
    title -- 图表标题 (默认为 'Integer Value Distribution')
    """
    # 1. 统计每个数值的出现次数
    from collections import defaultdict
    value_counts = defaultdict(int)
    for num in data:
        value_counts[num] += 1
    
    # 2. 准备绘图数据（按数值大小排序）
    sorted_values = sorted(value_counts.keys())
    counts = [value_counts[val] for val in sorted_values]
    
    # 3. 创建图表（核心修改：按分类显示，而非连续数值轴）
    plt.figure(figsize=(12, 6))
    
    # 生成类别标签（将数值转换为字符串）
    x_labels = [str(val) for val in sorted_values]
    
    # 绘图：X轴位置使用0,1,2,...索引（实现柱子紧挨显示）
    x_positions = range(len(sorted_values))
    bars = plt.bar(x_positions, counts, width=0.8, color='skyblue', edgecolor='black')
    
    # 在柱子上方添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.annotate(f'{count}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # 4. 设置X轴为分类轴（确保数值不连续时柱子紧挨）
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    
    # 5. 添加图表元素
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Integer Values', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    
    # 6. 自动调整布局并保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"📈 Chart saved to: {output_path}")
    
    # 7. 显示图表（可选）
    # plt.show()
    
    # 8. 返回统计数据
    return {"values": sorted_values, "counts": counts}



def visualize_resource_usage(data_dict, output_path="resource_usage.jpg"):
    """
    可视化资源使用情况并保存为JPG图片
    
    参数:
    data_dict -- 字典，键为类别，值为[计算量, 访存量]
    output_path -- 输出图片路径，默认为"resource_usage.jpg"
    """
    # 对类别进行排序
    sorted_keys = sorted(data_dict.keys())
    
    # 提取计算量和访存量
    computations = [data_dict[k][0] for k in sorted_keys]
    memory_accesses = [data_dict[k][1] for k in sorted_keys]
    
    # 计算总量和百分比
    total_comp = sum(computations)
    total_mem = sum(memory_accesses)
    
    comp_percentages = [comp/total_comp*100 for comp in computations]
    mem_percentages = [mem/total_mem*100 for mem in memory_accesses]
    
    # 设置图形布局
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置柱状图位置和宽度
    x = np.arange(len(sorted_keys))
    bar_width = 0.35
    
    # 绘制柱状图
    comp_bars = ax.bar(x - bar_width/2, computations, bar_width, 
                       color='skyblue', alpha=0.8, label='ops')
    mem_bars = ax.bar(x + bar_width/2, memory_accesses, bar_width, 
                      color='salmon', alpha=0.8, label='mem')
    
    # 在柱子上添加百分比标签
    for i, bar in enumerate(comp_bars):
        height = bar.get_height()
        ax.annotate(f'{comp_percentages[i]:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
    
    for i, bar in enumerate(mem_bars):
        height = bar.get_height()
        ax.annotate(f'{mem_percentages[i]:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
    
    # 添加标题和标签
    ax.set_title('ops and men distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('category', fontsize=12)
    ax.set_ylabel('percentage', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'category {k}' for k in sorted_keys])  # 添加"类别"前缀更清晰
    ax.legend(fontsize=10)
    
    # 设置Y轴网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整字体大小
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    
    # 保存为高质量JPG图片（300 DPI）
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format='jpeg')
    print(f"可视化结果已保存至: {output_path}")

def plot_pie_charts(data_dicts, titles, labels, output_path="pie_charts.jpg", colors=None):
    """
    绘制多个饼状图并保存为图片
    
    参数:
    data_dicts -- 字典列表，每个字典包含饼图数据 {类型: 值}
    titles -- 每个饼图的标题列表
    labels -- 每个饼图的类别标签（字典的键）
    output_path -- 输出图片路径 (默认: "pie_charts.jpg")
    colors -- 可选的颜色列表 (默认使用内置颜色)
    """
    # 计算需要多少张图片 (每张最多4个饼图)
    num_plots = len(data_dicts)
    num_figures = ceil(num_plots / 4)
    
    # 内置颜色方案
    if colors is None:
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', 
                  '#c2c2f0', '#ffb3e6', '#ff6666', '#c2f0c2']
    
    # 逐张图片生成
    for fig_idx in range(num_figures):
        plt.figure(figsize=(15, 10))
        
        # 计算当前图片包含的饼图索引
        start_idx = fig_idx * 4
        end_idx = min(start_idx + 4, num_plots)
        num_current_plots = end_idx - start_idx
        
        # 创建子图布局
        if num_current_plots <= 2:
            rows, cols = 1, num_current_plots
        else:
            rows, cols = 2, 2
        
        # 绘制每个饼图
        for i, plot_idx in enumerate(range(start_idx, end_idx)):
            ax = plt.subplot(rows, cols, i + 1)
            data_dict = data_dicts[plot_idx]
            title = titles[plot_idx]
            label = labels[plot_idx]
            
            # 准备数据
            keys = list(data_dict.keys())
            values = [data_dict[k] for k in keys]
            
            # 绘制饼图
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=keys if label else None, 
                autopct=lambda p: f'{p:.1f}%' if p >= 5 else '',
                startangle=90,
                colors=colors[:len(keys)]
            )
            
            # 设置标题和图例
            ax.set_title(title, fontsize=15, pad=10)
            plt.setp(autotexts, size=8, weight="bold")
            
            # 添加数据标签
            if label and len(keys) > 1:
                plt.legend(
                    wedges, 
                    [f"{k}: {v}" for k, v in data_dict.items()],
                    title=label,
                    loc="best",
                    fontsize=20
                )
        
        # 调整布局并保存
        plt.tight_layout(pad=3.0)
        output = output_path.replace(".jpg", f"_{fig_idx+1}.jpg") if num_figures > 1 else output_path
        plt.savefig(output, format='jpg', dpi=150)
        plt.close()



from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from math import ceil
from matplotlib.ticker import FuncFormatter
def prosocse_data(data: list, path: str="./output"):
    max_list = [] #用于存放MKN中的最大值
    min_list = [] #用于存放MKN中的最小值
    arithmetic_intensity = {} #将算子的计算强度按照20的步长分类,每一类计算总的计算量和内存访问量

    mem_type = {} #用于存放不同算子类型的内存访问量
    ops_type = {} #用于存放不同算子类型的计算量

    total_mem = 0 #总内存访问量
    total_ops = 0 #总计算量

    for item in data:
        if len(item.shape) > 0:
            try:
                min_list.append(min(item.shape))
                max_list.append(max(item.shape))
            except Exception as e:
                pp = 0
        ai_type = (int(item.arithmetic_intensity // 20))
        ai_record = arithmetic_intensity.get(ai_type, [0, 0])  # [ops, mem]
        ai_record[0] += item.ops
        ai_record[1] += item.mem
        arithmetic_intensity[ai_type] = ai_record

 
        type_name = item.type.name
        mem_type[type_name] = mem_type.get(type_name, 0) + item.mem
        ops_type[type_name] = ops_type.get(type_name, 0) + item.ops

    # 将矩阵乘法的max边绘制出来
    plot_int_distribution(max_list,path+"max.png","max")

    # 将矩阵乘法的min边绘制出来
    plot_int_distribution(min_list,path+"min.png","min")

    # 将各个算子的arithmetic_intensity和平均arithmetic_intensity绘制出来
    # 生成可视化图表
    visualize_resource_usage(arithmetic_intensity,output_path=path+"compute_memory_analysis.jpg")
  

    # 将mem_type和ops_type绘制出来
    titles = ["fops", "mem"]
    labels = ["ops_type", "mem_type"]
    # 生成饼图
    plot_pie_charts(
        data_dicts=[ops_type,mem_type],
        titles=titles,
        labels=labels,
        output_path=path+"multi_pie_charts.jpg"
    )
    return 0




model_shape = {
    "default":{
        "input": np.random.randn(1, 3, 224, 224).astype(np.float32),
        "other": np.array([1], dtype=np.int64)
    },
    "efficientnet":{
        "input": np.random.randn(10, 3, 240, 240).astype(np.float32),
        "other": np.array([1], dtype=np.int64)
    },
    "yolo":{
        "input": np.random.randn(1, 3, 640, 640).astype(np.float32),
        "other": np.array([1], dtype=np.int64)
    },
    "bev":{
        "input": np.random.randn(1, 6, 3, 480, 480).astype(np.float32),
        "other": np.array([1], dtype=np.int64)
    },
    "detr":{
        "input": np.random.randn(1, 3, 640, 640).astype(np.float32),
        "other": np.array([1], dtype=np.int64)
    },
}

import glob
import os
from common import jax_ops_mem as jax

base_path = "/data/coding/learnML/model_parse/"  # 替换为你的目录路径
# 示例使用
if __name__ == "__main__":
    # 设置目标目录路径
    directory_path = "/data/coding/learnML/model_parse/onnx/"  # 替换为你的目录路径

    # 获取所有以.onnx结尾的文件名
    onnx_files = [os.path.basename(f) for f in glob.glob(os.path.join(directory_path, "*.onnx"))]

    input_data ={}
    for item in onnx_files:
        if item.startswith("efficientnet"):
            input_data = model_shape["efficientnet"]
        elif item.startswith("yolo"):
            input_data = model_shape["yolo"]
        elif item.startswith("bev"):
            input_data = model_shape["bev"]
        elif item.startswith("detr"):
            input_data = model_shape["detr"]
        else:
            input_data = model_shape["default"]
        model_path = directory_path + item
        
        model_record = []
        # 获取算子详情
        try:
            details = get_operator_details(model_path, input_data)
            
            # 打印结果
            print("-"*30,model_path,"-"*30)
            print(f"\n获取到 {len(details)} 个算子详细信息")
            for i, op in enumerate(details):
                node_info = op["node"]
                op_type = node_info['op_type']
                attributes = op['attributes']
                input_list = []
                inputs = op['inputs']
                # pp = type(inputs)
                # ii = inputs.shape
                input_list.append(inputs)
                input_shape = []
                for name, data in inputs.items():
                    if data is not None:
                        input_shape.append(data.shape)
                        # print(f"  {name}: shape={data.shape}, dtype={data.dtype}")
                    else:
                        # print(f"  {name}: default")
                        continue
                op_record = jax.get_flops_and_memory_access(op_type,input_shape,attributes)
                if isinstance(op_record, jax.CalculateBlockInfo):
                    model_record.append(op_record)

            clean_name = item[:-5] if item.endswith('.onnx') else item
            output_path = base_path +"output/" + clean_name + "/"
            if not os.path.exists(output_path):
                # 创建目录（包括中间路径）
                os.makedirs(output_path)
            prosocse_data(model_record, path=output_path)


        
        except Exception as e:
            print("-"*50,model_path)
            logger.error(f"处理失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
