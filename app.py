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
    page_icon="🏮", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 核心 CSS 美化 (引入书法字体 + 纹理背景 + 渐变标题) ---
st.markdown("""
<style>
    /* 引入谷歌字体：马善政毛笔体 */
    @import url('https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

    /* 全局背景 - 增加极淡的波点纹理，拒绝死板的纯白 */
    .stApp {
        background-color: #fcfcfc;
        background-image: radial-gradient(#e5e7eb 1px, transparent 1px);
        background-size: 20px 20px; /* 波点间距 */
    }
    
    /* 顶部容器 */
    .header-container {
        text-align: center;
        padding: 40px 0;
        margin-bottom: 30px;
    }
    
    /* 标题样式升级 - 书法字体 + 红金渐变 */
    .app-title {
        font-family: 'Ma Shan Zheng', cursive; /* 使用书法字体 */
        font-size: 5rem;
        line-height: 1.2;
        /* 制作红金渐变文字 */
        background: linear-gradient(45deg, #c0392b 0%, #d35400 40%, #c0392b 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1); /* 轻微阴影增加立体感 */
        margin-bottom: 10px;
    }
    
    .app-subtitle {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.2rem;
        color: #555;
        letter-spacing: 4px; /* 增加字间距，显得大气 */
        border-top: 1px solid #ddd;
        border-bottom: 1px solid #ddd;
        display: inline-block;
        padding: 10px 30px;
        margin-top: 10px;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #eee;
    }

    /* 上传区域美化 - 增加活力 */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #b0b0b0;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 30px;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #c0392b; /* 悬停变红 */
        box-shadow: 0 4px 12px rgba(192, 57, 43, 0.1);
    }

    /* 结果卡片 - 悬浮效果 */
    .result-card {
        background: white;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #eee;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    /* 鼠标放上去会上浮 */
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }

    .landmark-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 2.2rem;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 15px;
        border-left: 5px solid #c0392b; /* 左侧红条装饰 */
        padding-left: 15px;
    }

    /* 标签美化 */
    .tag-span {
        display: inline-block;
        background: linear-gradient(to right, #fdfbfb, #ebedee); 
        border: 1px solid #ddd;
        color: #555;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-bottom: 20px;
    }
    
    /* 装饰性分割线 */
    hr {
        margin: 20px 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0));
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 详细文献资料库 ---
def get_landmark_info(name):
    """资料库逻辑"""
    info = {
        "title": "未知地标",
        "tags": ["待识别"],
        "desc": "暂无该地标的详细文献资料。",
        "history": "请确认模型分类是否正确。"
    }

    if "gutian" in name:
        info = {
            "title": "古田会议会址",
            "tags": ["福建上杭", "思想建党", "全国重点文物"],
            "desc": "古田会议会址，原名廖氏宗祠（万源祠），位于福建省龙岩市上杭县古田镇溪背村。始建于清道光二十八年（1848年），是一座典型的客家宗祠建筑。会址背靠参天古木的社下山，面朝视野开阔的田野。",
            "history": "1929年12月28日至29日，中国共产党红军第四军第九次代表大会在此召开。会议一致通过了毛泽东起草的《古田会议决议》。这次会议确立了“思想建党、政治建军”的原则，解决了如何把以农民为主要成分的军队建设成为无产阶级性质的新型人民军队这个根本性问题。"
        }
    elif "nanchang" in name:
        info = {
            "title": "南昌八一起义纪念馆",
            "tags": ["江西南昌", "军旗升起", "第一枪"],
            "desc": "南昌八一起义纪念馆位于江西省南昌市中山路，其旧址原为“江西大旅社”。这座中西合璧的灰色五层砖瓦楼房，曾是当时南昌城的最高建筑，1927年成为了起义的总指挥部。",
            "history": "1927年8月1日凌晨，在周恩来、贺龙、叶挺、朱德、刘伯承等人的领导下，南昌起义爆发。这打响了武装反抗国民党反动派的第一枪，标志着中国共产党独立领导革命战争、创建人民军队和武装夺取政权的开始。"
        }
    elif "jinggangshan" in name:
        info = {
            "title": "井冈山革命博物馆",
            "tags": ["江西井冈山", "革命摇篮", "天下第一山"],
            "desc": "井冈山被誉为“中国革命的摇篮”。井冈山革命博物馆依山而建，气势恢宏，全面展示了井冈山革命根据地的创立、发展和斗争历程。",
            "history": "1927年10月，毛泽东率领秋收起义部队到达井冈山，创立了中国第一个农村革命根据地。在这里，点燃了“工农武装割据”的星星之火，开辟了“农村包围城市、武装夺取政权”的中国革命特色道路。"
        }
    elif "zunyi" in name:
        info = {
            "title": "遵义会议会址",
            "tags": ["贵州遵义", "伟大转折", "中西合璧"],
            "desc": "遵义会议会址位于贵州省遵义市老城子尹路96号，原为黔军将领柏辉章的私邸。建筑为二层砖木结构，中西合璧，既有中式回廊，又有西式窗花。",
            "history": "1935年1月，中共中央政治局在此召开扩大会议。会议结束了“左”倾教条主义错误在中央的统治，确立了毛泽东在党和红军中的领导地位。在极端危急的历史关头，挽救了党，挽救了红军，挽救了中国革命。"
        }
    elif "luding" in name:
        info = {
            "title": "泸定桥",
            "tags": ["四川甘孜", "大渡河", "铁索桥"],
            "desc": "泸定桥始建于清康熙四十四年（1705年），横跨大渡河，是连接藏汉交通的咽喉要道。桥身由13根碗口粗的铁链组成。",
            "history": "1935年5月29日，中央红军红四团的22名勇士，冒着敌人的密集火力，攀踏着悬空的铁索，冲过大渡河，夺取了泸定桥。粉碎了蒋介石企图让红军成为“石达开第二”的梦想，为红军北上打开了通道。"
        }
    elif "monument" in name:
        info = {
            "title": "人民英雄纪念碑",
            "tags": ["北京天安门", "国家象征", "永垂不朽"],
            "desc": "位于北京天安门广场中心，是新中国为了纪念在人民解放战争和人民革命中牺牲的人民英雄而建立的。碑身正面镌刻着毛泽东题写的“人民英雄永垂不朽”八个镏金大字。",
            "history": "1949年9月30日奠基，1958年建成。碑座四周镶嵌着十幅巨大的汉白玉浮雕，概括了从1840年鸦片战争到1949年解放战争的一百多年间，中国人民反帝反封建的革命斗争史。"
        }
    elif "ccp" in name or "first_congress" in name:
        info = {
            "title": "中共一大会址",
            "tags": ["上海", "党的诞生地", "石库门"],
            "desc": "位于上海市兴业路76号，是一幢典型的上海石库门里弄建筑。1921年7月23日，中国共产党第一次全国代表大会在此召开。",
            "history": "这里是中国共产党的“产房”。大会通过了中国共产党第一个纲领和决议，选举产生了中央局。毛泽东曾说：“中国产生了共产党，这是开天辟地的大事变。”"
        }
    elif "tiananmen" in name:
        info = {
            "title": "天安门广场",
            "tags": ["北京", "城市中心", "开国大典"],
            "desc": "世界上最大的城市中心广场，位于北京市中心，南北长880米，东西宽500米，可容纳100万人举行盛大集会。",
            "history": "天安门广场记载了中国人民不屈不挠的革命精神。五四运动、一二·九运动、开国大典等重大历史事件均发生于此。它是新中国的象征。"
        }
    elif "xibaipo" in name:
        info = {
            "title": "西柏坡纪念馆",
            "tags": ["河北平山", "最后指挥所", "两个务必"],
            "desc": "位于河北省平山县，曾是中共中央所在地。党中央在这里指挥了震惊中外的辽沈、淮海、平津三大战役。",
            "history": "1949年3月，党的七届二中全会在此召开，毛泽东提出了著名的“两个务必”。周恩来评价：“西柏坡是毛主席和党中央进入北平，解放全中国的最后一个农村指挥所。”"
        }
    elif "baotashan" in name or "yanan" in name:
        info = {
            "title": "延安宝塔山",
            "tags": ["陕西延安", "精神灯塔", "岭山寺塔"],
            "desc": "宝塔山古称嘉岭山，位于延安城东南，延河之滨。山顶的唐代宝塔是延安城的标志性建筑。",
            "history": "“几回回梦里回延安，双手搂定宝塔山。”在革命战争年代，宝塔山是中国革命圣地的象征，是指引无数有志青年奔向光明的灯塔。中共中央在延安十三年，培育了光照千秋的延安精神。"
        }
    elif "nanhu" in name or "ship" in name:
        info = {
            "title": "嘉兴南湖红船",
            "tags": ["浙江嘉兴", "红船精神", "一大闭幕"],
            "desc": "停泊在浙江嘉兴南湖畔的一艘画舫。1921年8月初，中共一大从上海转移至此继续举行。",
            "history": "在这艘船上，中共一大闭幕，正式宣告了中国共产党的诞生。习近平总书记提出的“红船精神”——开天辟地、敢为人先的首创精神，坚定理想、百折不挠的奋斗精神，由此源起。"
        }
    elif "ruijin" in name:
        info = {
            "title": "瑞金革命遗址",
            "tags": ["江西瑞金", "红色故都", "苏维埃"],
            "desc": "瑞金是中华苏维埃共和国临时中央政府诞生地，被誉为“红色故都”。著名的“红井”和形似红军八角帽的大礼堂旧址皆坐落于此。",
            "history": "1931年11月7日，中华苏维埃共和国临时中央政府在瑞金宣告成立，这是中国共产党建立国家政权的首次尝试，为后来新中国的建立积累了宝贵的治国理政经验。"
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
    st.markdown("### ⚙️ 设置")
    password = st.text_input("管理员密码", type="password")
    st.markdown("---")
    st.caption("© 2025 一拍即知项目组")

if password != ADMIN_PASSWORD:
    # 极简锁定页
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 50vh;">
        <h2 style="color: #9CA3AF !important;">🔒</h2>
        <h3 style="color: #374151 !important;">ACCESS LOCKED</h3>
        <p style="color: #6B7280;">请在左侧栏输入密码解锁</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 6. 主界面构建 ---

# 顶部 Header (使用了书法字体)
st.markdown("""
<div class="header-container">
    <div class="app-title">一拍即知</div>

</div>
""", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("⚠️ 系统提示：模型文件未找到，请联系管理员。")
    st.stop()

# 宽屏布局
col_upload, col_display = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown("#### 📤 影像上传")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is None:
        st.markdown("""
        <div style="text-align: center; color: #9CA3AF; margin-top: 20px;">
            <p>等待图像输入...</p>
        </div>
        """, unsafe_allow_html=True)

# 识别逻辑
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # 进度条
    progress_bar = col_upload.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i + 1)
    
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

    # 右侧展示
    with col_display:
        st.markdown(f"""
        <div class="result-card">
            <div class="landmark-title">{info['title']}</div>
        """, unsafe_allow_html=True)
        
        # 标签
        tags_html = "".join([f'<span class="tag-span">#{tag}</span>' for tag in info['tags']])
        st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            # 图片加圆角
            st.image(image, caption="原始影像", use_column_width=True)
            st.markdown(f"**AI匹配度**：`{score:.2f}%`")
            
        with c2:
            st.markdown("#### 🏛️ 地标简介")
            st.write(info['desc'])
            st.markdown("#### 📜 历史文脉")
            st.write(info['history'])
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # 欢迎页
    with col_display:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <h3 style="color: #2c3e50;">欢迎使用一拍即知</h3>
            <p style="color: #7f8c8d; max-width: 500px; margin: 0 auto;">
                基于深度学习的智能识别系统。<br>
                请在左侧上传照片，即刻获取地标背后的历史故事。
            </p>
        </div>
        """, unsafe_allow_html=True)
