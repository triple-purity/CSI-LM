import gradio as gr
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import torch
from transformers import AutoConfig

from sklearn.preprocessing import MinMaxScaler
from models.StuModels import TimeModule

from utils.preprocess_tools import calculate_csi_ratio, hampel_filter, phase_calibration, resample_csi_sequence, get_doppler_spectrum
from dataset.datasets import data_norm

# --------------------------
# 1. 模型设置
# 加载本地模型权重
# --------------------------
llama_names = ['unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B']
gpt_names = ['openai-community/gpt2']

llm_name = 'openai-community/gpt2'
llm_config = AutoConfig.from_pretrained(llm_name)
d_llm = llm_config.hidden_size if llm_name in llama_names else llm_config.n_embd

dfs_model = TimeModule(
    class_num=6,
    input_dim=121,
    token_kernels=[5, 11, 21],
    llm_name='openai-community/gpt2',
    d_model=1024,
    embed_size=d_llm,
    n_heads=8,
    num_encoder=6,
)
# model.load_state_dict(torch.load("time_series_model.pth"))
dfs_model.eval()

# 设置数据参数
csi_ratio_unified_len = 600
dfs_unified_len = 1800

# --------------------------
# 2. 数据导入函数
# --------------------------
def load_and_visualize(file):
    """加载时间序列文件并绘制原始数据"""
    mat = loadmat(file.name)
    csi_data = mat['csi_data']
    csi_data = csi_data.reshape(csi_data.shape[0], 3, -1)
    abs_csi = np.abs(csi_data)

    plt.figure(figsize=(10, 4))
    for i in range(0,5):
        plt.plot(abs_csi[:,0,i])
    plt.title("Raw CSI Data")
    plt.xlabel("Time")
    plt.ylabel("CSI Amplitude")
    fig = plt.gcf()
    plt.close(fig)
    return csi_data, fig

# --------------------------
# 3. 数据预处理函数
# --------------------------
def preprocess_data(csi_data, method):
    """预处理数据"""
    if method == "CSI-Ratio Phase":
        csi_ratio, antenna_index = calculate_csi_ratio(csi_data)
        csi_ratio = np.concatenate((csi_ratio[:, :antenna_index, :], csi_ratio[:, antenna_index+1:, :]), axis=1)
        angle_csi_ratio = np.angle(csi_ratio)
        angle_csi_ratio = phase_calibration(hampel_filter(angle_csi_ratio))
        resample_csi_ratio = resample_csi_sequence(angle_csi_ratio, target_length=csi_ratio_unified_len)

        plt.figure(figsize=(10, 4))
        for i in range(0,30):
            plt.plot(resample_csi_ratio[:,0,i])
        plt.title("CSI-Ratio Phase")
        plt.xlabel("Time")
        plt.ylabel("Phase")
        fig = plt.gcf()
        plt.close(fig)

        tensor_csi_ratio = torch.tensor(resample_csi_ratio, dtype=torch.float32)
        tensor_csi_ratio = data_norm(tensor_csi_ratio, norm_type='mean_std')
        return tensor_csi_ratio, fig
    elif method == "DFS":
        dfs, t, f = get_doppler_spectrum(csi_data)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(dfs), shading='gouraud', cmap='jet')
        plt.title('Doppler Frequency Shift', fontsize=14)
        plt.ylabel('Frequency', fontsize=12)
        plt.xlabel('Time', fontsize=12)
        plt.colorbar(label='Magnitude')
        fig = plt.gcf()
        plt.close(fig)

        _, sample_index = dfs.shape
        if sample_index >= dfs_unified_len:
            doppler_spectrum = dfs[:, :dfs_unified_len]
        else:
            doppler_spectrum = np.concatenate([dfs, np.zeros((dfs.shape[0], dfs_unified_len-sample_index))], axis=1)
        tensor_dfs = torch.tensor(doppler_spectrum, dtype=torch.float32)
        tensor_dfs = tensor_dfs.permute(1, 0)
        return tensor_dfs, fig
    else:
        raise ValueError("Invalid method selected.")

# --------------------------
# 4. 模型预测分类
# --------------------------
def predict(data):
    """使用模型预测/分类"""
    input_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1)  # shape: [1, seq_len, 1]
    with torch.no_grad():
        output = dfs_model(input_tensor)
    prob = torch.softmax(output, dim=1).numpy()[0]

     # 绘制横向柱状图
    classes = [f"Class {i}" for i in range(len(prob))]
    plt.figure(figsize=(8, 4))
    plt.barh(classes, prob, color='skyblue')
    plt.xlabel('Probability')
    plt.title('Classification Probabilities')
    fig = plt.gcf()
    plt.close(fig)
    return fig # 返回分类概率图片

# --------------------------
# 5. Gradio界面构建
# --------------------------
with gr.Blocks(title="基于预训练大模型的手势识别系统") as app:
    gr.Markdown("## 🙋‍♂️手势识别工具")
    
    with gr.Tab("1. 数据加载"):
        with gr.Row():
            file_input = gr.File(label="上传CSI文件(mat文件)")
            load_btn = gr.Button("加载数据")
        raw_plot = gr.Plot(label="原始数据可视化")
        raw_data = gr.State() # 隐藏传递数据
    
    with gr.Tab("2. 数据处理"):
        method_select = gr.Radio(choices=["CSI-Ratio Phase", "DFS"], label="选择处理方式", value="CSI-Ratio Phase")
        process_btn = gr.Button("处理数据")
        processed_plot = gr.Plot(label="处理后的数据") 
        processed_data = gr.State()
    
    with gr.Tab("3. 模型预测"):
        predict_btn = gr.Button("运行预测")
        label_output = gr.Label(label="分类概率")

    # --------------------------
    # 4. 事件绑定
    # --------------------------
    load_btn.click(
        fn=load_and_visualize,
        inputs=file_input,
        outputs=[raw_data, raw_plot]
    )
    
    process_btn.click(
        fn=preprocess_data,
        inputs=[raw_data, method_select],
        outputs=[processed_data, processed_plot]
    )
    
    predict_btn.click(
        fn=predict,
        inputs=processed_data,
        outputs=label_output
    )

# --------------------------
# 5. 启动应用
# --------------------------
if __name__ == "__main__":
    app.launch(share=False)  # 设置 share=True 生成临时公共链接
