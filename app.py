import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# --- 1. 页面基础配置 (简洁大方风格) ---
st.set_page_config(
    page_title="一拍即知 | 智能地标百科",
    page_icon="🏛️", # 改用博物馆图标，更显文化感
    layout="wide",   # 改为宽屏模式，更大气
    initial_sidebar_state="collapsed" # 默认收起侧边栏，突出主界面
)

# --- 2. 核心 CSS 美化 (极简现代风) ---
st.markdown("""
<style>
    /* 全局字体与背景 - 极简白灰 */
    .stApp {
        background-color: #F9FAFB; /* 极淡的灰白背景，护眼且高级 */
        font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    
    /* 文字颜色控制 - 深灰为主，易于阅读 */
    h1, h2, h3, h4, h5, h6 {
        color: #1F2937 !important; /* 接近黑色的深灰 */
        font-weight: 700;
    }
    p, div, li {
        color: #4B5563; /* 次深灰正文 */
        line-height: 1.7; /* 增加行高，提升阅读感 */
    }

    /* 顶部标题区 - 现代排版 */
    .header-container {
        text-align: center;
        padding: 60px 0 40px 0;
        background: white;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 40px;
    }
    .app-title {
        font-size: 3.5rem;
        color: #111827;
        letter-spacing: -1px;
        margin-bottom: 10px;
        background: -webkit-linear-gradient(45deg, #111827, #4B5563);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* 侧边栏 - 纯净白 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    /* 上传区域美化 - 虚线框 */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #D1D5DB;
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        transition: border 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #3B82F6; /* 悬停变为科技蓝 */
    }

    /* 结果展示区 - 杂志风格 */
    .result-section {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-top: 20px;
    }
    .landmark-name {
        font-size: 2.2rem;
        color: #111827;
        border-left: 6px solid #111827; /* 黑色竖条装饰 */
        padding-left: 20px;
        margin-bottom: 20px;
    }
    .info-tag {
        display: inline-block;
        background-color: #F3F4F6;
        color: #374151;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin-right: 10px;
        margin-bottom: 20px;
    }
    
    /* 按钮样式 - 黑色极简 */
    div.stButton > button {
        background-color: #111827;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #374151;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 详细文献资料库 (Detailed Knowledge Base) ---
def get_landmark_info(name):
    """根据类别名称返回详细的文献资料"""
    info = {
        "title": "未知地标",
        "tags": ["待录入"],
        "desc": "暂无该地标的详细文献资料。",
        "history": "请确认模型分类是否正确。"
    }

    # 关键词匹配逻辑 (根据你的数据集类别)
    if "gutian" in name:
        info = {
            "title": "古田会议会址",
            "tags": ["福建上杭", "建党建军", "全国重点文物"],
            "desc": "古田会议会址，原名廖氏宗祠（万源祠），位于福建省龙岩市上杭县古田镇溪背村。始建于清道光二十八年（1848年），是一座典型的客家宗祠建筑。会址背靠参天古木的社下山，面朝视野开阔的田野。",
            "history": """
            **历史转折：** 1929年12月28日至29日，中国共产党红军第四军第九次代表大会（即古田会议）在此召开。毛泽东同志主持会议并作了重要报告。
            
            **核心意义：** 会议一致通过了毛泽东起草的《中国共产党红军第四军第九次代表大会决议案》（即《古田会议决议》）。这次会议确立了“思想建党、政治建军”的原则，解决了如何把以农民为主要成分的军队建设成为无产阶级性质的新型人民军队这个根本性问题。古田会议不仅是红军建设史上的里程碑，也是中国共产党历史上的重要篇章。
            """
        }
    elif "nanchang" in name:
        info = {
            "title": "南昌八一起义纪念馆",
            "tags": ["江西南昌", "军旗升起", "第一枪"],
            "desc": "南昌八一起义纪念馆位于江西省南昌市中山路，其旧址原为“江西大旅社”。这座中西合璧的灰色五层砖瓦楼房，曾是当时南昌城的最高建筑。1927年，这里成为了起义的总指挥部。",
            "history": """
            **石破天惊：** 1927年8月1日凌晨，在周恩来、贺龙、叶挺、朱德、刘伯承等人的领导下，南昌起义爆发。这打响了武装反抗国民党反动派的第一枪，标志着中国共产党独立领导革命战争、创建人民军队和武装夺取政权的开始。
            
            **历史地位：** 8月1日因此被定为中国人民解放军的建军节。纪念馆内珍藏了大量珍贵的革命文物，详细还原了起义前后的历史细节，是感受人民军队诞生历程的核心地标。
            """
        }
    elif "jinggangshan" in name:
        info = {
            "title": "井冈山革命博物馆",
            "tags": ["江西井冈山", "革命摇篮", "农村包围城市"],
            "desc": "井冈山，被誉为“中国革命的摇篮”。井冈山革命博物馆依山而建，气势恢宏，全面展示了井冈山革命根据地的创立、发展和斗争历程。",
            "history": """
            **道路开辟：** 1927年10月，毛泽东率领秋收起义部队到达井冈山，创立了中国第一个农村革命根据地。在这里，点燃了“工农武装割据”的星星之火，开辟了“农村包围城市、武装夺取政权”的中国革命特色道路。
            
            **艰苦卓绝：** 著名的“黄洋界保卫战”、“朱德的扁担”等故事皆发生于此。井冈山精神——坚定信念、艰苦奋斗、实事求是、敢闯新路、依靠群众、勇于胜利，成为了中国共产党精神谱系的重要组成部分。
            """
        }
    elif "zunyi" in name:
        info = {
            "title": "遵义会议会址",
            "tags": ["贵州遵义", "生死转折", "中西合璧"],
            "desc": "遵义会议会址位于贵州省遵义市老城子尹路96号，原为黔军将领柏辉章的私邸。建筑为二层砖木结构，中西合璧，既有中式回廊，又有西式窗花。",
            "history": """
            **伟大转折：** 1935年1月15日至17日，中共中央政治局在此召开扩大会议。会议结束了“左”倾教条主义错误在中央的统治，确立了毛泽东在党和红军中的领导地位。
            
            **关键时刻：** 遵义会议在极端危急的历史关头，挽救了党，挽救了红军，挽救了中国革命，是党的历史上一个生死攸关的转折点，标志着中国共产党在政治上开始走向成熟。
            """
        }
    elif "luding" in name:
        info = {
            "title": "泸定桥",
            "tags": ["四川甘孜", "长征", "铁索桥"],
            "desc": "泸定桥始建于清康熙四十四年（1705年），横跨大渡河，是连接藏汉交通的咽喉要道。桥身由13根碗口粗的铁链组成，其中9根作底链，4根分两侧作扶手。",
            "history": """
            **飞夺天险：** 1935年5月29日，中央红军红四团的22名勇士，冒着敌人的密集火力，攀踏着悬空的铁索，冲过大渡河，夺取了泸定桥。
            
            **奇迹：** “飞夺泸定桥”是长征中最为惊心动魄的战役之一，粉碎了蒋介石企图让红军成为“石达开第二”的梦想，为红军北上打开了通道。
            """
        }
    elif "monument" in name:
        info = {
            "title": "人民英雄纪念碑",
            "tags": ["北京天安门", "国家象征", "永垂不朽"],
            "desc": "位于北京天安门广场中心，是新中国为了纪念在人民解放战争和人民革命中牺牲的人民英雄而建立的。碑身正面镌刻着毛泽东题写的“人民英雄永垂不朽”八个镏金大字。",
            "history": """
            **历史铭记：** 1949年9月30日奠基，1958年建成。碑座四周镶嵌着十幅巨大的汉白玉浮雕，概括了从1840年鸦片战争到1949年解放战争的一百多年间，中国人民反帝反封建的革命斗争史。
            """
        }
    elif "ccp" in name or "first_congress" in name:
        info = {
            "title": "中共一大会址",
            "tags": ["上海兴业路", "党的诞生地", "石库门"],
            "desc": "位于上海市兴业路76号（原望志路106号），是一幢典型的上海石库门里弄建筑。1921年7月23日，中国共产党第一次全国代表大会在此召开。",
            "history": """
            **开天辟地：** 大会通过了中国共产党第一个纲领和决议，选举产生了中央局。毛泽东曾说：“中国产生了共产党，这是开天辟地的大事变。”这里是中国共产党的“产房”，也是中国共产党人的精神家园。
            """
        }
    elif "tiananmen" in name:
        info = {
            "title": "天安门广场",
            "tags": ["北京", "世界最大", "国家庆典"],
            "desc": "世界上最大的城市中心广场，位于北京市中心，南北长880米，东西宽500米，可容纳100万人举行盛大集会。",
            "history": """
            **见证历史：** 天安门广场记载了中国人民不屈不挠的革命精神和大无畏的英雄气概。五四运动、一二·九运动、开国大典等重大历史事件均发生于此。它是新中国的象征，也是全国各族人民向往的地方。
            """
        }
    elif "xibaipo" in name:
        info = {
            "title": "西柏坡纪念馆",
            "tags": ["河北平山", "最后一个农村指挥所", "赶考"],
            "desc": "位于河北省平山县，曾是中共中央所在地。党中央在这里指挥了震惊中外的辽沈、淮海、平津三大战役。",
            "history": """
            **新中国前夜：** 1949年3月，党的七届二中全会在此召开，毛泽东提出了著名的“两个务必”。周恩来曾评价：“西柏坡是毛主席和党中央进入北平，解放全中国的最后一个农村指挥所。”
            """
        }
    elif "baotashan" in name or "yanan" in name:
        info = {
            "title": "延安宝塔山",
            "tags": ["陕西延安", "精神灯塔", "革命圣地"],
            "desc": "宝塔山古称嘉岭山，位于延安城东南，延河之滨。山顶的唐代宝塔（岭山寺塔）高44米，是延安城的标志性建筑。",
            "history": """
            **精神象征：** “几回回梦里回延安，双手搂定宝塔山。”在革命战争年代，宝塔山是中国革命圣地的象征，是指引无数有志青年奔向光明的灯塔。中共中央在延安十三年，培育了光照千秋的延安精神。
            """
        }
    elif "nanhu" in name or "ship" in name:
        info = {
            "title": "嘉兴南湖红船",
            "tags": ["浙江嘉兴", "红船精神", "一大闭幕"],
            "desc": "停泊在浙江嘉兴南湖畔的一艘画舫（丝网船）。1921年8月初，中共一大从上海转移至此继续举行。",
            "history": """
            **启航：** 在这艘船上，中共一大闭幕，正式宣告了中国共产党的诞生。这艘船因此被称为“红船”。习近平总书记提出的“红船精神”——开天辟地、敢为人先的首创精神，坚定理想、百折不挠的奋斗精神，立党为公、忠诚为民的奉献精神，由此源起。
            """
        }
    elif "ruijin" in name:
        info = {
            "title": "瑞金革命遗址",
            "tags": ["江西瑞金", "红都", "苏维埃共和国"],
            "desc": "瑞金是中华苏维埃共和国临时中央政府诞生地，被誉为“红色故都”。著名的“红井”和形似红军八角帽的大礼堂旧址皆坐落于此。",
            "history": """
            **共和国摇篮：** 1931年11月7日，中华苏维埃共和国临时中央政府在瑞金宣告成立，这是中国共产党建立国家政权的首次尝试，为后来新中国的建立积累了宝贵的治国理政经验。
            """
        }
    
    return info

# --- 4. 资源加载与预处理 ---
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

# --- 5. 密码锁逻辑 (侧边栏隐藏式) ---
ADMIN_PASSWORD = "123"
with st.sidebar:
    st.markdown("### ⚙️ 设置")
    password = st.text_input("管理员密码", type="password")
    st.markdown("---")
    st.caption("© 2025 一拍即知项目组")

if password != ADMIN_PASSWORD:
    # 锁定界面：极简风格
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh;">
        <h2 style="color: #9CA3AF !important;">🔒</h2>
        <h3 style="color: #374151 !important;">系统访问受限</h3>
        <p style="color: #6B7280;">请在左侧栏输入密码以解锁内容</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 6. 主界面构建 ---

# 顶部 Header
st.markdown("""
<div class="header-container">
    <div class="app-title">一拍即知</div>
   
</div>
""", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("⚠️ 系统提示：模型文件未找到，请联系管理员。")
    st.stop()

# 布局：宽屏两列布局
col_upload, col_display = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown("#### 📤 影像上传")
    st.info("请上传JPG/PNG图片，系统将自动进行AI分析。")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    # 占位图或默认提示
    if uploaded_file is None:
        st.markdown("""
        <div style="margin-top: 20px; text-align: center; color: #9CA3AF;">
            <p>Waiting for upload...</p>
        </div>
        """, unsafe_allow_html=True)

# 识别逻辑
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # 显示加载条
    progress_bar = col_upload.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i + 1)
    
    # 推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 1)
        
    class_id = predicted.item()
    raw_name = class_names[class_id]
    score = confidence.item() * 100
    
    # 获取详细资料
    info = get_landmark_info(raw_name)

    # --- 右侧展示区 ---
    with col_display:
        st.markdown(f"""
        <div class="result-section">
            <div class="landmark-name">{info['title']}</div>
        """, unsafe_allow_html=True)
        
        # 标签展示
        tags_html = "".join([f'<span class="info-tag">#{tag}</span>' for tag in info['tags']])
        st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
        
        # 左右分栏：左图右文（在卡片内部）
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.image(image, caption="原始影像", use_column_width=True)
            st.markdown(f"**AI置信度**：`{score:.2f}%`")
            
        with c2:
            st.markdown("#### 🏛️ 地标简介")
            st.write(info['desc'])
            
            st.markdown("#### 📜 历史文脉")
            st.markdown(info['history'])
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # 欢迎页/空状态 (在右侧)
    with col_display:
        st.markdown("""
        <div style="padding: 40px; border-radius: 16px; background-color: white; border: 1px solid #E5E7EB; text-align: center;">
            <h3 style="color: #374151;">欢迎使用一拍即知</h3>
            <p style="color: #6B7280; max-width: 500px; margin: 0 auto;">
                这是一个基于深度学习的智能地标识别系统。请在左侧上传您拍摄的红色地标照片，
                我们将为您提供详尽的历史文献资料与背景解读。
            </p>
            <br>
            <p style="font-size: 0.9rem; color: #9CA3AF;">支持设备：电脑 / 平板 / 手机</p>
        </div>
        """, unsafe_allow_html=True)
