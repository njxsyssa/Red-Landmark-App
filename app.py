import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="一拍即知 | Snap & Know",
    page_icon="🍎", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 核心 CSS 美化 (修复可见性与渲染问题) ---
st.markdown("""
<style>
    /* 引入字体 */
    @import url('https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=Noto+Serif+SC:wght@500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* 全局背景 - 极简灰白格调 */
    .stApp {
        background-color: #F5F5F7; /* Apple 官网常用的浅灰 */
        background-image: radial-gradient(#d1d1d1 1px, transparent 1px);
        background-size: 30px 30px;
    }

    /* 顶部标题区 */
    .header-container {
        text-align: center;
        padding: 60px 0 40px 0;
    }
    
    .app-title {
        font-family: 'Ma Shan Zheng', cursive;
        font-size: 5.5rem;
        background: linear-gradient(135deg, #1d1d1f 0%, #424245 100%); /* Apple深空灰渐变 */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -2px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* === 🍎 Apple Style Slogan === */
    .app-subtitle {
        font-family: 'Inter', 'Noto Serif SC', sans-serif;
        color: #86868b; /* Apple 经典的次级文本灰 */
        font-size: 1.2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        margin-top: 5px;
    }
    .highlight-text {
        color: #1d1d1f;
        font-weight: 600;
    }

    /* === 核心修复：上传区域文字颜色 === */
    /* 强制指定 Upload 组件的 label 颜色 */
    div[data-testid="stFileUploader"] label {
        color: #1d1d1f !important;
        font-size: 1.1rem !important;
        font-family: 'Noto Serif SC', serif !important;
        font-weight: bold !important;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #d2d2d7;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 18px; /* 更圆润的圆角 */
        padding: 30px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #0071e3; /* Apple Blue */
        background-color: #ffffff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }

    /* === 核心修复：结果卡片样式 === */
    .info-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08); /* 柔和的高级阴影 */
        border: 1px solid rgba(0,0,0,0.02);
        color: #1d1d1f; /* 强制深色字 */
    }

    .card-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 2.8rem;
        color: #1d1d1f;
        margin-bottom: 20px;
        font-weight: 700;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 20px;
    }

    .section-head {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1d1d1f;
        margin-top: 30px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .section-head::before {
        content: '';
        display: inline-block;
        width: 6px;
        height: 24px;
        background-color: #c0392b; /* 唯一的红色点缀，克制 */
        margin-right: 12px;
        border-radius: 3px;
    }

    .card-text {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #424245; /* 正文深灰 */
        font-family: sans-serif;
    }

    .tag-pill {
        display: inline-block;
        background-color: #f5f5f7;
        color: #1d1d1f;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-right: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 详细文献资料库 ---
def get_landmark_info(name):
    info = {
        "title": "未能识别",
        "tags": ["无匹配数据"],
        "desc": "无法匹配到当前的数据库信息。",
        "history": "请尝试上传更清晰的照片。"
    }

    if "gutian" in name:
        info = {
            "title": "古田会议会址",
            "tags": ["福建上杭", "建党原则", "1929"],
            "desc": "古田会议会址，原名廖氏宗祠，位于福建龙岩上杭县。始建于清道光年间，是一座典型的客家宗祠建筑。背靠参天古木，面朝广阔田野，庄重而古朴。",
            "history": "1929年12月，红四军第九次代表大会在此召开。会议通过了《古田会议决议》，确立了“思想建党、政治建军”的原则。这是人民军队建设史上的里程碑，解决了将以农民为主要成分的军队建设成无产阶级性质新型军队的根本问题。"
        }
    elif "nanchang" in name:
        info = {
            "title": "南昌起义纪念馆",
            "tags": ["江西南昌", "第一枪", "建军节"],
            "desc": "旧址原为“江西大旅社”，位于南昌中山路。这座中西合璧的灰色五层砖瓦楼房是当时南昌的最高建筑。1927年，这里成为了起义的总指挥部。",
            "history": "1927年8月1日，在周恩来、贺龙、叶挺等领导下，南昌起义爆发，打响了武装反抗国民党反动派的第一枪。它标志着中国共产党独立领导革命战争、创建人民军队的开始。8月1日因此被定为中国人民解放军建军节。"
        }
    elif "jinggangshan" in name:
        info = {
            "title": "井冈山革命博物馆",
            "tags": ["革命摇篮", "天下第一山", "江西"],
            "desc": "井冈山被誉为“中国革命的摇篮”。博物馆依山而建，气势恢宏，全景式展示了井冈山革命根据地的斗争历史与精神风貌。",
            "history": "1927年10月，毛泽东率领秋收起义部队到达井冈山，创立了中国第一个农村革命根据地。在这里，点燃了“工农武装割据”的星星之火，开辟了“农村包围城市、武装夺取政权”的中国革命特色道路。"
        }
    elif "zunyi" in name:
        info = {
            "title": "遵义会议会址",
            "tags": ["贵州遵义", "伟大转折", "1935"],
            "desc": "位于贵州遵义子尹路，原为黔军将领私邸。建筑为二层砖木结构，融合了中式回廊与西式窗花风格，是当年红军长征途中占领的唯一一座城市里最好的建筑。",
            "history": "1935年1月，中共中央政治局在此召开扩大会议。会议结束了“左”倾教条主义错误，确立了毛泽东在党和红军中的领导地位。在极端危急关头，挽救了党、红军和中国革命，是党的历史上生死攸关的转折点。"
        }
    elif "luding" in name:
        info = {
            "title": "泸定桥",
            "tags": ["大渡河", "铁索桥", "22勇士"],
            "desc": "始建于清康熙年间，横跨汹涌的大渡河，由13根碗口粗的铁链组成。它是连接藏汉交通的咽喉要道，地势险要，素有“十三根铁链劈开大渡河”之说。",
            "history": "1935年5月29日，中央红军红四团22名勇士，冒着密集火力攀踏悬空铁索，飞夺泸定桥。这一奇迹粉碎了蒋介石让红军成为“石达开第二”的企图，为红军北上打开了生命通道。"
        }
    elif "monument" in name:
        info = {
            "title": "人民英雄纪念碑",
            "tags": ["北京天安门", "国家象征", "永垂不朽"],
            "desc": "矗立于天安门广场中心，是新中国为纪念牺牲的人民英雄而建。碑身正面镌刻毛泽东题写的“人民英雄永垂不朽”八个镏金大字，背面为周恩来题写的碑文。",
            "history": "1949年奠基，1958年建成。碑座十幅巨大汉白玉浮雕，概括了从1840年鸦片战争到1949年解放战争百年来，中国人民不屈不挠的革命斗争史。它是中华民族精神的丰碑。"
        }
    elif "ccp" in name or "first_congress" in name:
        info = {
            "title": "中共一大会址",
            "tags": ["上海", "党的诞生", "石库门"],
            "desc": "位于上海兴业路76号，典型的石库门里弄建筑。1921年7月23日，中国共产党第一次全国代表大会在此秘密召开。",
            "history": "这里是中国共产党的“产房”。大会通过了第一个纲领，选举产生了中央局。毛泽东称之为“开天辟地的大事变”。中国革命的面貌从此焕然一新。"
        }
    elif "tiananmen" in name:
        info = {
            "title": "天安门广场",
            "tags": ["北京中心", "开国大典", "世界最大"],
            "desc": "世界上最大的城市中心广场，南北长880米，东西宽500米，可容纳百万群众集会。城楼庄严肃穆，是中华人民共和国的象征。",
            "history": "1949年10月1日，毛泽东在此庄严宣告中华人民共和国成立。五四运动、一二·九运动等重大历史事件均发生于此。它见证了中国人民从站起来到富起来、强起来的伟大飞跃。"
        }
    elif "xibaipo" in name:
        info = {
            "title": "西柏坡纪念馆",
            "tags": ["河北平山", "最后指挥所", "两个务必"],
            "desc": "位于河北平山县，依山傍水。曾是中共中央所在地，在这里指挥了震惊中外的辽沈、淮海、平津三大战役。",
            "history": "1949年3月，七届二中全会在此召开，毛泽东提出了著名的“两个务必”。周恩来评价其为“毛主席和党中央进入北平，解放全中国的最后一个农村指挥所”。"
        }
    elif "baotashan" in name or "yanan" in name:
        info = {
            "title": "延安宝塔山",
            "tags": ["陕西延安", "精神灯塔", "古塔"],
            "desc": "宝塔山古称嘉岭山，位于延河之滨。山顶的唐代宝塔高44米，是历史名城延安的标志性建筑。",
            "history": "“几回回梦里回延安，双手搂定宝塔山。”在革命战争年代，宝塔山是革命圣地的象征，是指引无数有志青年奔向光明的灯塔。中共中央在延安十三年，培育了光照千秋的延安精神。"
        }
    elif "nanhu" in name or "ship" in name:
        info = {
            "title": "嘉兴南湖红船",
            "tags": ["浙江嘉兴", "红船精神", "一大闭幕"],
            "desc": "停泊在嘉兴南湖的一艘画舫。1921年8月初，因上海会场受袭，中共一大转移至此继续举行。",
            "history": "在这艘船上，中共一大闭幕，正式宣告中国共产党诞生。习近平总书记提出的“红船精神”——开天辟地、敢为人先的首创精神，坚定理想、百折不挠的奋斗精神，立党为公、忠诚为民的奉献精神，由此源起。"
        }
    elif "ruijin" in name:
        info = {
            "title": "瑞金革命遗址",
            "tags": ["江西瑞金", "红色故都", "苏维埃"],
            "desc": "瑞金是中华苏维埃共和国临时中央政府诞生地。叶坪、沙洲坝等旧址群保存完好，红井依然清冽。",
            "history": "1931年11月，中华苏维埃共和国临时中央政府在此成立，是中国共产党建立国家政权的首次尝试。这里被誉为“共和国摇篮”，为新中国的建立积累了宝贵的治国理政经验。"
        }
    
    return info

# --- 4. 资源加载 ---
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

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. 密码锁逻辑 ---
ADMIN_PASSWORD = "123"
with st.sidebar:
    st.markdown("### ⚙️ Admin Access")
    password = st.text_input("Password", type="password")
    st.markdown("---")
    st.caption("© 2025 一拍即知")

if password != ADMIN_PASSWORD:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #86868b;">
        <h2>🔒 Locked</h2>
        <p>Please verify your identity in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 6. 主界面构建 ---

# 顶部 Header
st.markdown("""
<div class="header-container">
    <div class="app-title">一拍即知</div>
    <div class="app-subtitle">
        <span style="margin: 0 10px; color: #d2d2d7;">|</span> 让每一张照片，都有故事。
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("⚠️ 系统提示：模型文件未找到，请联系管理员。")
    st.stop()

# 宽屏布局：左3（上传+图），右4（信息卡片）
col_left, col_right = st.columns([3, 4], gap="large")

with col_left:
    st.markdown("##### 📤 上传影像 / Upload")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # 图片样式
        st.markdown("""
        <style>
            img { 
                border-radius: 12px; 
                box-shadow: 0 8px 24px rgba(0,0,0,0.1); 
                transition: transform 0.3s;
            }
            img:hover { transform: scale(1.01); }
        </style>
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    else:
        st.info("👈 请点击上方区域选择照片")

# 识别与展示逻辑
if uploaded_file is not None:
    # 进度条
    progress_bar = col_left.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i + 1)
    progress_bar.empty()
    
    # 推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    class_id = predicted.item()
    raw_name = class_names[class_id]
    score = confidence.item() * 100
    
    # 获取资料
    info = get_landmark_info(raw_name)

    # --- 右侧信息展示区 (修复HTML渲染) ---
    with col_right:
        # 我们不再使用一个巨大的 f-string，而是分段渲染，这样更稳定
        
        # 1. 标题与置信度
        st.markdown(f"""
        <div class="info-card">
            <div class="card-title">{info['title']}</div>
            <div style="margin-bottom: 25px;">
                {''.join([f'<span class="tag-pill">{tag}</span>' for tag in info['tags']])}
                <span style="float: right; color: #86868b; font-size: 0.9rem; margin-top: 5px;">
                    AI Match: <b>{score:.1f}%</b>
                </span>
            </div>
            
            <div class="section-head">地标简介</div>
            <p class="card-text">{info['desc']}</p>
            
            <div class="section-head">历史文脉</div>
            <p class="card-text">{info['history']}</p>
        </div>
        """, unsafe_allow_html=True)
