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

csi_ratio_model = TimeModule(
    class_num=6,
    input_dim=30,
    token_kernels=[5, 11, 21],
    time_stride=4,
    llm_name='openai-community/gpt2',
    d_model=1024,
    embed_size=d_llm,
    n_heads=8,
    num_encoder=6,
)

dfs_model = TimeModule(
    class_num=6,
    input_dim=30,
    token_kernels=[5, 11, 21],
    time_stride=10,
    llm_name='openai-community/gpt2',
    d_model=1024,
    embed_size=d_llm,
    n_heads=8,
    num_encoder=6,
)

# model.load_state_dict(torch.load("time_series_model.pth"))
csi_ratio_model.eval()
dfs_model.eval()

# 设置数据参数
csi_ratio_unified_len = 600
dfs_unified_len = 1800

# 手势图片对应
gesture_names = {0: ['Push&Pull', './images/push.png'], 1: ['Sweep', './images/sweep.png'], 
                 2: ['Clap', './images/sweep.png'], 3: ['Silde', './images/Side.png'], 
                 4: ['Draw-O', './images/draw-o.png'], 5: ['Draw-ZigZag', './images/draw-zigzag.png']}


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
def preprocess_data(csi_data, method, progress=gr.Progress()):
    """预处理数据"""
    progress(0, desc="🔄 正在初始化...")

    if method == "CSI-Ratio Phase":
        progress(0.2, desc="📡 计算 CSI Ratio...")
        csi_ratio, antenna_index = calculate_csi_ratio(csi_data)
        csi_ratio = np.concatenate((csi_ratio[:, :antenna_index, :], csi_ratio[:, antenna_index+1:, :]), axis=1)
        
        progress(0.4, desc="📐 相位校准中...")
        angle_csi_ratio = np.angle(csi_ratio)
        angle_csi_ratio = phase_calibration(hampel_filter(angle_csi_ratio))
        
        progress(0.6, desc="🔁 重采样...")
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
        tensor_csi_ratio = tensor_csi_ratio.permute(1, 0, 2)

        progress(1.0, desc="✅ 完成 CSI-Ratio 处理")
        return tensor_csi_ratio, fig
    elif method == "DFS":
        progress(0.2, desc="📡 计算多普勒频谱...")
        dfs, t, f = get_doppler_spectrum(csi_data)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(dfs), shading='gouraud', cmap='jet')
        plt.title('Doppler Frequency Shift', fontsize=14)
        plt.ylabel('Frequency', fontsize=12)
        plt.xlabel('Time', fontsize=12)
        plt.colorbar(label='Magnitude')
        fig = plt.gcf()
        plt.close(fig)

        progress(0.6, desc="🔁 补零/截断...")
        _, sample_index = dfs.shape
        if sample_index >= dfs_unified_len:
            doppler_spectrum = dfs[:, :dfs_unified_len]
        else:
            doppler_spectrum = np.concatenate([dfs, np.zeros((dfs.shape[0], dfs_unified_len-sample_index))], axis=1)
        tensor_dfs = torch.tensor(doppler_spectrum, dtype=torch.float32)
        tensor_dfs = tensor_dfs.permute(1, 0)
        tensor_dfs = tensor_dfs.unsqueeze(0)

        progress(1.0, desc="✅ 完成 DFS 处理")
        return tensor_dfs, fig
    else:
        raise ValueError("Invalid method selected.")

# --------------------------
# 4. 模型预测分类
# --------------------------
def predict(data, method, progress=gr.Progress()):
    """使用模型预测/分类"""
    progress(0, desc="🔍 正在进行手势识别...")

    input_tensor = data
    with torch.no_grad():
        if method == "CSI-Ratio Phase":
            action_logits, _= csi_ratio_model.predict(input_tensor, decoder_mask=True)
            action_probs = torch.mean(torch.softmax(action_logits, dim=-1), dim=0).numpy()[0]
            action_index = torch.argmax(action_prob)
            progress(0.5, desc="🧠 分析中...")
        else:
            action_logits, action_index = csi_ratio_model.predict(input_tensor, decoder_mask=True)
            action_probs = torch.softmax(action_logits, dim=1).numpy()[0]
            progress(0.5, desc="🧠 分析中...")

    action_prob = action_probs[action_index]
    action_label = gesture_names[action_index][0]
    action_fig_path = gesture_names[action_index][1]
    
    # 加载图像并绘制
    img = plt.imread(action_fig_path)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    fig = plt.gcf()
    plt.close(fig)

    result_text = f"🎯 识别结果：{action_label}\n📈 置信度：{action_probs[action_index]:.2%}"
    progress(1.0, desc="✅ 完成")
    return result_text, fig 

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
        with gr.Row():
            method_select = gr.Radio(
                choices=["CSI-Ratio Phase", "DFS"],
                label="选择处理方式",
                value="CSI-Ratio Phase",
                elem_classes=["method-radio"]
            )

        process_btn = gr.Button("⚙️ 开始处理数据", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**处理进度**")
            progress_bar = gr.Progress()
            """
            status_text = gr.Textbox(
                label="状态信息",
                value="等待开始处理...",
                lines=3,
                interactive=False,
                elem_classes=["status-box"]
            )"""
        with gr.Column(scale=1):
            processed_plot = gr.Plot(label="📊 处理后的数据可视化", elem_id="plot-area")

        processed_data = gr.State()

        # 自定义CSS样式
        app.css += """
            .method-radio {
                font-weight: bold;
            }

            .status-box {
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 10px;
                font-size: 15px;
                animation: fadeIn 0.6s ease-in-out;
            }

            .gr-button {
                transition: all 0.3s ease;
            }

            .gr-button:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        """
    
    with gr.Tab("3. 模型预测"):
        predict_btn = gr.Button("🧠 运行预测", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**预测进度**")
            progress_bar = gr.Progress()
            result_text = gr.Textbox(
                label="预测结果",
                value="等待预测...",
                lines=4,
                interactive=False,
                elem_classes=["result-box"]
            )
        with gr.Column(scale=1):
            image_output = gr.Image(label="🤞 手势示意图", interactive=False)

        # 自定义CSS样式
        app.css += """
            .result-box {
                font-size: 16px;
                font-weight: bold;
                background-color: #f0f8ff;
                border: 2px solid #87ceeb;
                padding: 10px;
                color: #333;
                animation: fadeIn 1s ease-in-out;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        """
    
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
        inputs=[processed_data, method_select],
        outputs=[result_text, image_output]
    )

# --------------------------
# 5. 启动应用
# --------------------------
if __name__ == "__main__":
    app.launch(share=False)  # 设置 share=True 生成临时公共链接
