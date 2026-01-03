import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="一拍即知 | 红色地标智能百科",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 核心 CSS 美化 (强制高对比度 + 博物馆卡片风) ---
st.markdown("""
<style>
    /* 引入字体 */
    @import url('https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=Noto+Serif+SC:wght@500;700&display=swap');

    /* 全局背景 - 暖米色，护眼且有质感 */
    .stApp {
        background-color: #f4f4f0; 
        background-image: linear-gradient(to right, #e0e0e0 1px, transparent 1px),
                          linear-gradient(to bottom, #e0e0e0 1px, transparent 1px);
        background-size: 40px 40px; /* 网格纹理 */
    }

    /* 顶部标题区 */
    .header-container {
        text-align: center;
        padding: 40px 0;
        margin-bottom: 20px;
        background: transparent;
    }
    
    .app-title {
        font-family: 'Ma Shan Zheng', cursive;
        font-size: 4.5rem;
        /* 深青色渐变，沉稳大气 */
        background: linear-gradient(to right, #0F2027, #203A43, #2C5364); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-shadow: 2px 2px 0px rgba(255,255,255,0.5);
    }
    
    .app-subtitle {
        font-family: 'Noto Serif SC', serif;
        color: #555;
        font-size: 1.1rem;
        letter-spacing: 3px;
        border-top: 1px solid #999;
        border-bottom: 1px solid #999;
        display: inline-block;
        padding: 8px 20px;
        margin-top: 10px;
    }

    /* === 核心修复：内容卡片容器 === */
    /* 这是一个纯白色的盒子，用来放文字，保证对比度 */
    .info-card-container {
        background-color: #FFFFFF !important; /* 强制纯白背景 */
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); /* 悬浮阴影 */
        height: 100%; /* 填满高度 */
    }

    /* === 核心修复：强制文字颜色 === */
    /* 无论系统主题如何，强制这里面的字是深色的 */
    .info-card-container h2, .info-card-container h3, .info-card-container h4 {
        color: #1a1a1a !important; /* 纯黑标题 */
        font-family: 'Noto Serif SC', serif;
    }
    
    .info-card-container p, .info-card-container li, .info-card-container span {
        color: #333333 !important; /* 深灰正文 */
        line-height: 1.8; /* 增加行距，利于阅读 */
        font-size: 16px;
    }

    /* 装饰性标题：带竖线的标题 */
    .section-header {
        border-left: 5px solid #2C5364; /*由于青色竖条 */
        padding-left: 15px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1.3rem;
        background: linear-gradient(90deg, #f0f0f0 0%, transparent 100%); /* 渐变背景条 */
        padding-top: 5px;
        padding-bottom: 5px;
    }

    /* 标签样式 */
    .tag-box {
        display: inline-block;
        background-color: #eef2f3;
        color: #2c3e50 !important;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 8px;
        border: 1px solid #d1d5db;
    }

    /* 上传框美化 */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #a0a0a0;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 详细文献资料库 ---
def get_landmark_info(name):
    """资料库逻辑 - 保持丰富内容"""
    info = {
        "title": "未能识别",
        "tags": ["无匹配数据"],
        "desc": "无法匹配到当前的数据库信息。",
        "history": "请尝试上传更清晰的照片。"
    }

    if "gutian" in name:
        info = {
            "title": "古田会议会址",
            "tags": ["福建上杭", "思想建党", "1929年"],
            "desc": "古田会议会址，原名廖氏宗祠（万源祠），位于福建省龙岩市上杭县古田镇。始建于清道光二十八年（1848年），是一座典型的客家宗祠建筑。会址依山而建，庄重古朴。",
            "history": "1929年12月，红四军第九次代表大会在此召开。会议通过了毛泽东起草的《古田会议决议》，确立了“思想建党、政治建军”的原则。这是中国共产党和红军建设史上的重要里程碑，解决了如何把以农民为主要成分的军队建设成为无产阶级性质的新型人民军队的根本问题。"
        }
    elif "nanchang" in name:
        info = {
            "title": "南昌八一起义纪念馆",
            "tags": ["江西南昌", "军旗升起", "第一枪"],
            "desc": "旧址原为“江西大旅社”，位于南昌市中山路。这座灰色五层砖瓦楼房是当时南昌的最高建筑，中西合璧风格。1927年，这里成为起义的总指挥部。",
            "history": "1927年8月1日，在周恩来、贺龙、叶挺、朱德、刘伯承等领导下，南昌起义爆发，打响了武装反抗国民党反动派的第一枪。它标志着中国共产党独立领导革命战争、创建人民军队和武装夺取政权的开始。8月1日也因此被定为建军节。"
        }
    elif "jinggangshan" in name:
        info = {
            "title": "井冈山革命博物馆",
            "tags": ["江西井冈山", "革命摇篮", "农村包围城市"],
            "desc": "井冈山被誉为“中国革命的摇篮”。博物馆依山而建，气势恢宏，全景式展示了井冈山革命根据地的斗争历史。",
            "history": "1927年10月，毛泽东率领秋收起义部队到达井冈山，创立了中国第一个农村革命根据地。在这里，中国共产党开辟了“农村包围城市、武装夺取政权”的道路，点燃了工农武装割据的星星之火。"
        }
    elif "zunyi" in name:
        info = {
            "title": "遵义会议会址",
            "tags": ["贵州遵义", "生死转折", "中西合璧"],
            "desc": "位于贵州省遵义市子尹路，原为黔军将领柏辉章私邸。建筑为二层砖木结构，融合了中式回廊与西式窗花风格。",
            "history": "1935年1月，中共中央政治局在此召开扩大会议。会议结束了“左”倾教条主义错误，确立了毛泽东在党和红军中的领导地位。在极端危急关头，挽救了党、红军和中国革命，是党的历史上生死攸关的转折点。"
        }
    elif "luding" in name:
        info = {
            "title": "泸定桥",
            "tags": ["四川甘孜", "大渡河", "铁索桥"],
            "desc": "始建于清康熙年间，横跨大渡河，由13根碗口粗的铁链组成。它是连接藏汉交通的咽喉要道，地势险要。",
            "history": "1935年5月29日，中央红军红四团22名勇士，冒着密集火力攀踏悬空铁索，飞夺泸定桥。这一奇迹粉碎了蒋介石让红军成为“石达开第二”的企图，为红军北上打开了生命通道。"
        }
    elif "monument" in name:
        info = {
            "title": "人民英雄纪念碑",
            "tags": ["北京天安门", "国家象征", "永垂不朽"],
            "desc": "矗立于天安门广场中心，是新中国为纪念牺牲的人民英雄而建。碑身正面镌刻毛泽东题写的“人民英雄永垂不朽”八个镏金大字。",
            "history": "1949年奠基，1958年建成。碑座十幅汉白玉浮雕，概括了从1840年鸦片战争到1949年解放战争百年来，中国人民反帝反封建的革命斗争史。它是中华民族精神的丰碑。"
        }
    elif "ccp" in name or "first_congress" in name:
        info = {
            "title": "中共一大会址",
            "tags": ["上海", "党的诞生地", "石库门"],
            "desc": "位于上海兴业路76号，典型的石库门建筑。1921年7月23日，中国共产党第一次全国代表大会在此召开。",
            "history": "这里是中国共产党的“产房”。大会通过了第一个纲领，选举产生了中央局。毛泽东称之为“开天辟地的大事变”。中国革命的面貌从此焕然一新。"
        }
    elif "tiananmen" in name:
        info = {
            "title": "天安门广场",
            "tags": ["北京", "世界最大", "开国大典"],
            "desc": "世界上最大的城市中心广场，南北长880米，东西宽500米，可容纳百万群众集会。城楼庄严肃穆，是国家象征。",
            "history": "1949年10月1日，毛泽东在此庄严宣告中华人民共和国成立。五四运动、一二·九运动等重大历史事件均发生于此。它见证了中国人民不屈不挠的革命精神。"
        }
    elif "xibaipo" in name:
        info = {
            "title": "西柏坡纪念馆",
            "tags": ["河北平山", "最后指挥所", "两个务必"],
            "desc": "位于河北平山县，依山傍水。曾是中共中央所在地，指挥了辽沈、淮海、平津三大战役。",
            "history": "1949年3月，七届二中全会在此召开，毛泽东提出了著名的“两个务必”。周恩来评价其为“毛主席和党中央进入北平，解放全中国的最后一个农村指挥所”。"
        }
    elif "baotashan" in name or "yanan" in name:
        info = {
            "title": "延安宝塔山",
            "tags": ["陕西延安", "精神灯塔", "岭山寺塔"],
            "desc": "宝塔山古称嘉岭山，位于延河之滨。山顶的唐代宝塔高44米，是历史名城延安的标志。",
            "history": "在革命战争年代，宝塔山是革命圣地的象征，是指引无数有志青年奔向光明的灯塔。中共中央在延安十三年，领导了抗日战争和解放战争，培育了光照千秋的延安精神。"
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
            "desc": "瑞金是中华苏维埃共和国临时中央政府诞生地。叶坪、沙洲坝等旧址群保存完好。",
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
    st.markdown("### ⚙️ 管理员设置")
    password = st.text_input("请输入访问密码", type="password")
    st.markdown("---")
    st.caption("© 2025 一拍即知项目组")

if password != ADMIN_PASSWORD:
    st.markdown("""
    <div style="text-align: center; margin-top: 100px; color: #666;">
        <h2>🔒 系统已锁定</h2>
        <p>请在左侧侧边栏输入密码以继续访问。</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 6. 主界面构建 ---

# 顶部 Header
st.markdown("""
<div class="header-container">
    <div class="app-title">一拍即知</div>
    <div class="app-subtitle">SNAP & KNOW · 红色地标智能百科</div>
</div>
""", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("⚠️ 模型文件加载失败，请联系管理员。")
    st.stop()

# 宽屏布局：左3（上传+图），右4（信息卡片）
col_left, col_right = st.columns([3, 4], gap="large")

with col_left:
    st.markdown("##### 📤 上传影像")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # 图片加阴影和圆角
        st.markdown("""
        <style>
            img { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        </style>
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    else:
        # 空状态占位
        st.info("👈 请上传图片开始识别")

# 识别与展示逻辑
if uploaded_file is not None:
    # 进度条
    progress_bar = col_left.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i + 1)
    progress_bar.empty() # 跑完消失
    
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

    # --- 右侧信息展示区 (放入白色卡片容器) ---
    with col_right:
        # 使用 div class="info-card-container" 包裹所有内容
        st.markdown(f"""
        <div class="info-card-container">
            <h2 style="font-size: 2.5rem; margin-bottom: 10px; border-bottom: 2px solid #eee; padding-bottom: 15px;">
                {info['title']}
            </h2>
            
            <div style="margin-bottom: 25px;">
                {''.join([f'<span class="tag-box">{tag}</span>' for tag in info['tags']])}
                <span style="float: right; color: #aaa; font-size: 0.9rem;">AI置信度: {score:.1f}%</span>
            </div>
            
            <div class="section-header">🏛️ 地标简介</div>
            <p>{info['desc']}</p>
            
            <div class="section-header">📜 历史文脉</div>
            <p>{info['history']}</p>
            
        </div>
        """, unsafe_allow_html=True)
