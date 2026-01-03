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

# --- 2. 核心 CSS 美化 (高对比度 + 红金主题) ---
st.markdown("""
<style>
    /* ================== 全局强制配色 ================== */
    /* 强制主区域背景为极淡的米白色，模拟纸张，确保深色文字清晰可见 */
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* 强制主区域的所有文字颜色为深灰/黑色，解决看不清的问题 */
    .stApp p, .stApp div, .stApp label, .stApp li {
        color: #333333 !important;
    }
    
    /* 强制各级标题颜色为“中国红” */
    h1, h2, h3, h4, h5, h6 {
        color: #8B0000 !important;
        font-family: 'SimHei', 'Heiti SC', sans-serif; /* 尝试使用黑体，更庄重 */
    }

    /* ================== 顶部大标题 Banner ================== */
    /* 创建一个红底金边的横幅容器 */
    .main-header-container {
        background: linear-gradient(180deg, #A40000 0%, #8B0000 100%);
        padding: 30px 20px;
        border-radius: 8px;
        border: 2px solid #FFD700; /* 金色边框 */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        text-align: center;
    }
    
    .main-header-title {
        color: #FFD700 !important; /* 金色文字 */
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: 5px;
        text-shadow: 2px 2px 0px #000; /* 文字阴影增加立体感 */
    }
    
    .main-header-subtitle {
        color: #FFFFFF !important;
        font-size: 1.2rem;
        margin-top: 10px;
        opacity: 0.9;
    }

    /* ================== 侧边栏 (沉浸式深红) ================== */
    section[data-testid="stSidebar"] {
        background-color: #720000; /* 比主标题更深一点的红 */
        border-right: 2px solid #FFD700;
    }
    
    /* 侧边栏里的所有文字强制为金色或白色 */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #FFD700 !important; /* 金色 */
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important; /* 标题白 */
    }
    
    /* 侧边栏输入框特殊处理，防止文字看不清 */
    section[data-testid="stSidebar"] input {
        color: #333333 !important;
    }

    /* ================== 组件样式优化 ================== */
    /* 按钮样式 - 红底金字 */
    div.stButton > button {
        background: linear-gradient(to bottom, #B22222, #8B0000);
        color: #FFD700 !important;
        border: 1px solid #FFD700;
        font-weight: bold;
        font-size: 18px;
        padding: 10px 25px;
    }
    div.stButton > button:hover {
        background: #FF0000;
        border-color: #FFF;
        color: #FFF !important;
    }

    /* 结果展示卡片 */
    .result-box {
        background-color: #FFFFFF;
        border: 2px solid #8B0000;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
    }
    
    /* 进度条颜色 */
    div[data-testid="stProgress"] > div > div > div {
        background-color: #d4af37; /* 金色进度条 */
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 密码锁逻辑 ---
ADMIN_PASSWORD = "123" 

with st.sidebar:
    st.markdown("## 🚩 功能菜单")
    st.markdown("此系统专为红色地标识别设计。")
    st.markdown("---")
    password = st.text_input("🔓 管理员/演示密码", type="password")
    st.markdown("---")
    st.markdown("Designed by 颜笑组")

if password != ADMIN_PASSWORD:
    # 锁定界面也做一个简单的美化
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; padding: 40px; border: 2px dashed #8B0000; background-color: #fff0f0;">
            <h2 style="color: #8B0000;">⛔ 系统已锁定</h2>
            <p style="color: #555;">请输入密码以访问“一拍即知”系统。</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.stop()

# --- 4. 加载资源 ---
@st.cache_resource
def load_resources():
    class_names = []
    if os.path.exists("class_names.txt"):
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
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

# 预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. 主界面内容 ---

# 顶部 Banner
st.markdown("""
<div class="main-header-container">
    <div class="main-header-title">一拍即知</div>
    <div class="main-header-subtitle"></div>
</div>
""", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("❌ 模型加载失败，请检查服务器文件。")
    st.stop()

# 上传区
uploaded_file = st.file_uploader("📸 请上传或拍摄景点照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner('🚩 正在匹配红色数据库...'):
        time.sleep(0.8) 
        image = Image.open(uploaded_file).convert('RGB')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 1)
            
        class_id = predicted.item()
        result_name = class_names[class_id]
        score = confidence.item() * 100

    # --- 6. 结果展示区 ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_img, col_info = st.columns([1, 1.2])
    
    with col_img:
        # 给图片加个红框，更有感觉
        st.image(image, caption='待识别图像', use_column_width=True)
    
    with col_info:
        # 动态判定置信度颜色
        if score > 80:
            status_color = "#228B22" # 森林绿
            status_text = "识别成功"
        else:
            status_color = "#FF4500" # 橙红
            status_text = "可疑匹配"

        # 结果卡片
        st.markdown(f"""
        <div class="result-box">
            <h3 style="margin-top:0; color: #8B0000 !important; border-bottom: 2px solid #eee; padding-bottom: 10px;">
                🚩 识别结果：{result_name}
            </h3>
            <p style="font-size: 16px; margin-top: 15px;">
                <strong>匹配状态：</strong><span style="color:{status_color}; font-weight:bold;">{status_text}</span>
            </p>
            <p style="font-size: 16px;">
                <strong>AI置信度：</strong>{score:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(score))
        
    # --- 历史资料库 ---
    st.markdown("### 📜 历史背景资料")
    st.markdown("---")
    
    # 这里的内容可以根据你的PPT进一步充实
    desc_text = "该地标是重要的爱国主义教育基地，见证了中国革命的光辉历程。"
    
    if "gutian" in result_name:
        desc_text = """**古田会议会址**：1929年12月，中国工农红军第四军第九次党的代表大会在此召开。会议确立了“思想建党、政治建军”的原则，是人民军队建设史上的重要里程碑。"""
    elif "nanchang" in result_name:
        desc_text = """**南昌八一起义纪念馆**：1927年8月1日，中国共产党在这里打响了武装反抗国民党反动派的第一枪，宣告了中国共产党把中国革命进行到底的坚定立场，标志着中国共产党独立领导革命战争、创建人民军队和武装夺取政权的开始。"""
    elif "jinggangshan" in result_name:
        desc_text = """**井冈山革命根据地**：中国革命的摇篮。在这里，中国共产党开辟了“农村包围城市、武装夺取政权”的道路。"""
    elif "zunyi" in result_name:
        desc_text = """**遵义会议会址**：1935年1月在此召开的会议，结束了“左”倾教条主义错误在中央的统治，确立了毛泽东在党和红军中的领导地位，是党的历史上一个生死攸关的转折点。"""
    
    st.info(desc_text)

else:
    # 引导页
    st.markdown("""
    <div style="text-align: center; padding: 40px; color: #666; background-color: #fff; border-radius: 10px; border: 1px dashed #ccc;">
        <h4>👈 请在上方上传图片</h4>
        <p>系统将自动识别红色地标并关联历史背景</p>
    </div>
    """, unsafe_allow_html=True)
