import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler

# --------------------------
# 1. 模拟训练好的模型（替换为你的实际模型）
# --------------------------
class TimeSeriesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = torch.nn.Linear(32, 2)  # 假设是二分类任务
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))

# 加载本地模型权重（示例路径，替换为你的模型路径）
model = TimeSeriesModel()
# model.load_state_dict(torch.load("time_series_model.pth"))
model.eval()

# --------------------------
# 2. 数据处理函数
# --------------------------
def load_and_visualize(file):
    """加载时间序列文件并绘制原始数据"""
    df = pd.read_csv(file.name, header=None)  # 假设无表头
    plt.figure(figsize=(10, 4))
    plt.plot(df.values)
    plt.title("Raw Time Series Data")
    plt.close()
    return df.values, plt.gcf()

def preprocess_data(data, smooth_window=5, normalize=True):
    """数据预处理：滑动平均 + 归一化"""
    processed = pd.Series(data.flatten()).rolling(smooth_window).mean().values
    if normalize:
        processed = MinMaxScaler().fit_transform(processed.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(10, 4))
    plt.plot(processed)
    plt.title("Processed Data")
    plt.close()
    return processed, plt.gcf()

def predict(data):
    """使用模型预测/分类"""
    input_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1)  # shape: [1, seq_len, 1]
    with torch.no_grad():
        output = model(input_tensor)
    prob = torch.softmax(output, dim=1).numpy()[0]
    return {"Class 0": prob[0], "Class 1": prob[1]}  # 返回分类概率

# --------------------------
# 3. Gradio界面构建
# --------------------------
with gr.Blocks(title="时间序列分析系统") as app:
    gr.Markdown("## 🕒 时间序列分析工具")
    
    with gr.Tab("1. 数据加载"):
        with gr.Row():
            file_input = gr.File(label="上传时间序列文件（CSV/TXT）")
            load_btn = gr.Button("加载数据")
        raw_plot = gr.Plot(label="原始数据可视化")
        raw_data = gr.Dataframe(visible=False)  # 隐藏传递数据
    
    with gr.Tab("2. 数据处理"):
        smooth_slider = gr.Slider(1, 20, value=5, label="滑动窗口大小")
        normalize_check = gr.Checkbox(value=True, label="归一化数据")
        process_btn = gr.Button("处理数据")
        processed_plot = gr.Plot(label="处理后的数据") 
        processed_data = gr.Dataframe(visible=False)
    
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
        inputs=[raw_data, smooth_slider, normalize_check],
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
    app.launch(share=True)  # 设置 share=True 生成临时公共链接
