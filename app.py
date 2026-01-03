import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="一拍即知",
    page_icon="🇨🇳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. 核心 CSS 美化 (这是变好看的关键) ---
st.markdown("""
<style>
    /* 全局背景色 - 羊皮纸淡黄，保护视力且有历史感 */
    .stApp {
        background-image: linear-gradient(to bottom, #fdfbf7, #f3e5d8);
    }

    /* 顶部标题样式 - 红色渐变 + 阴影 */
    h1 {
        color: #8B0000;
        font-family: 'Kaiti', 'STKaiti', 'KaiTi', serif; /* 尝试调用楷体 */
        text-align: center;
        font-size: 3.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        border-bottom: 3px solid #d4af37; /* 金色下划线 */
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    
    /* 二级标题样式 */
    h2, h3 {
        color: #5c0e0e;
        font-family: 'Songti SC', 'SimSun', serif;
        text-align: center;
    }

    /* 侧边栏样式优化 */
    section[data-testid="stSidebar"] {
        background-color: #8B0000; /* 深红背景 */
        color: #FFD700; /* 金色文字 */
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    /* 按钮样式 - 红色按钮，金色文字 */
    div.stButton > button {
        background-color: #8B0000;
        color: white;
        border-radius: 10px;
        border: 2px solid #d4af37;
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #a80000;
        border-color: #ffd700;
        transform: scale(1.05);
    }

    /* 结果展示卡片 */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 10px solid #8B0000;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    
    .result-title {
        color: #8B0000;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .result-desc {
        color: #333;
        line-height: 1.6;
        font-size: 16px;
    }
    
    /* 上传区域边框 */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #8B0000;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 密码锁逻辑 (保留你的需求) ---
ADMIN_PASSWORD = "123" 

# 侧边栏装饰
with st.sidebar:
    st.markdown("## 🚩 红色足迹导航")
    st.markdown("---")
    st.info("欢迎使用红色地标智能识别系统。")
    password = st.text_input("🔑 请输入访问密码", type="password")

if password != ADMIN_PASSWORD:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.warning("⚠️ 系统已锁定，请输入密码后使用。")
    st.stop()

# --- 4. 加载资源 (模型与标签) ---
@st.cache_resource
def load_resources():
    class_names = []
    if os.path.exists("class_names.txt"):
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    # 确保这里的分类数量与 class_names 长度一致，如果报错请检查 class_names.txt
    if class_names:
        model.fc = nn.Linear(num_ftrs, len(class_names))
    
    try:
        model.load_state_dict(torch.load('red_landmark_model.pth', map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        return None, None
        
    return model, class_names

model, class_names = load_resources()

# 定义预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. 主界面内容 ---

# 标题栏
st.markdown("<h1>🌟 红色记忆 · 智能博览 🌟</h1>", unsafe_allow_html=True)
st.markdown("### —— 基于深度学习的极端场景地标识别系统 ——")
st.markdown("<br>", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("❌ 系统初始化失败：未找到模型文件或类别文件。请检查服务器配置。")
    st.stop()

# 布局：左边上传，右边展示介绍（或者居中）
col1, col2 = st.columns([1, 8]) # 简单的居中布局
with col2:
    uploaded_file = st.file_uploader("📷 点击上传或拍摄红色景点照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 显示加载动画
    with st.spinner('🚀 正在进行神经网络分析，请稍候...'):
        time.sleep(1) # 模拟一下分析过程，显得更有科技感
        image = Image.open(uploaded_file).convert('RGB')
        
        # 推理
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 1)
            
        class_id = predicted.item()
        result_name = class_names[class_id]
        score = confidence.item() * 100

    # --- 6. 结果展示区 (精心设计) ---
    st.markdown("---")
    
    # 使用两列布局：左图右文
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.image(image, caption='您上传的影像', use_column_width=True)
    
    with c2:
        # 判断置信度，给出不同的反馈
        if score > 80:
            emoji_icon = "🎯"
            color_bar = "green"
        elif score > 50:
            emoji_icon = "🤔"
            color_bar = "orange"
        else:
            emoji_icon = "❓"
            color_bar = "red"
            
        # 使用 HTML 卡片展示结果
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">{emoji_icon} 识别结果：{result_name}</div>
            <div style="color: grey; font-size: 14px;">AI 置信度：{score:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 进度条展示置信度
        st.progress(int(score))
        
        # 历史背景资料 (模拟数据，你可以根据需要扩展)
        st.markdown("#### 📖 历史回响")
        bg_info = "暂无详细资料"
        if "gutian" in result_name:
            bg_info = "1929年12月，中国工农红军第四军第九次党的代表大会在此召开，确立了“思想建党、政治建军”的原则。"
        elif "nanchang" in result_name:
            bg_info = "1927年8月1日，这里打响了武装反抗国民党反动派的第一枪，标志着中国共产党独立领导革命战争、创建人民军队的开始。"
        elif "jinggangshan" in result_name:
            bg_info = "井冈山，中国革命的摇篮。在这里，中国共产党开辟了“农村包围城市、武装夺取政权”的道路。"
        elif "zunyi" in result_name:
            bg_info = "遵义会议，是中共中央政治局的一次扩大会议。它在极端危急的历史关头，挽救了党，挽救了红军，挽救了中国革命。"
        elif "luding" in result_name:
            bg_info = "飞夺泸定桥，是中国工农红军长征中的一场重要战役。红军战士在昼夜急行军240里后，攀踏着铁索悬空的桥板，冲过火网。"
        
        st.info(bg_info)

else:
    # 没上传图片时的占位符，保持页面不空
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <h3>👈 请在上方上传图片开启体验</h3>
        <p>支持 JPG / PNG 格式</p>
    </div>
    """, unsafe_allow_html=True)

# 底部版权
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: grey;'>© 2025 红色地标识别课题组 | Powered by PyTorch & Streamlit</div>", unsafe_allow_html=True)
