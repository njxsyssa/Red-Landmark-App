import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
import textwrap  # 用于修复缩进导致的渲染 bug

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="一拍即知 | Snap & Know",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 核心 CSS 美化 (Ultimate Fix) ---
st.markdown("""
<style>
    /* 引入字体：马善政(标题) + 思源宋体(正文) + Inter(英文) */
    @import url('https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=Noto+Serif+SC:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* 全局背景 - 极简高级灰 */
    .stApp {
        background-color: #f5f5f7;
        background-image: radial-gradient(#d2d2d7 1px, transparent 1px);
        background-size: 24px 24px;
    }

    /* === Header: Apple Style === */
    .header-container {
        text-align: center;
        padding: 60px 0 40px 0;
    }
    .app-title {
        font-family: 'Ma Shan Zheng', cursive;
        font-size: 5rem;
        background: linear-gradient(135deg, #1d1d1f 0%, #434344 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .app-subtitle {
        font-family: 'Inter', 'Noto Serif SC', sans-serif;
        font-size: 1.25rem;
        color: #86868b;
        font-weight: 400;
        letter-spacing: 0.8px;
    }

    /* === Fix: 上传区域文字强制黑色 === */
    div[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.9); /* 半透明白底 */
        border: 2px dashed #a1a1a6;
        border-radius: 18px;
        padding: 40px 20px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #0071e3;
        box-shadow: 0 8px 20px rgba(0, 113, 227, 0.15);
    }
    /* 暴力强制所有子元素文字颜色 */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] div,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {
        color: #1d1d1f !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="stFileUploader"] label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* === Fix: 结果卡片样式 (无代码感) === */
    .info-card {
        background-color: #ffffff;
        border-radius: 24px;
        padding: 40px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        height: 100%;
    }
    .landmark-header {
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 20px;
        margin-bottom: 25px;
    }
    .landmark-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 2.5rem;
        color: #1d1d1f;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .tag-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .tag-pill {
        background-color: #f5f5f7;
        color: #1d1d1f;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    .match-score {
        background-color: #e8f2ff;
        color: #0071e3;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    /* 章节标题 */
    .section-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.2rem;
        color: #1d1d1f;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
    .section-title::before {
        content: '';
        display: block;
        width: 4px;
        height: 18px;
        background-color: #c0392b; /* 极简红条 */
        margin-right: 10px;
        border-radius: 2px;
    }
    
    /* 正文文字 */
    .section-content {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 1.05rem;
        line-height: 1.75;
        color: #424245;
        text-align: justify;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 深度文献资料库 (Detailed Database) ---
def get_landmark_info(name):
    """
    返回结构化数据：标题、标签列表、简介HTML、历史HTML、精神HTML
    """
    default_info = {
        "title": "未知地标",
        "tags": ["待识别"],
        "desc": "暂无详细资料，请确认图片是否清晰。",
        "history": "请尝试重新拍摄或上传。",
        "spirit": ""
    }
    
    # 数据库内容
    db = {
        "gutian": {
            "title": "古田会议会址",
            "tags": ["福建上杭", "思想建党", "1929年", "全国重点文物"],
            "desc": "古田会议会址，原名廖氏宗祠（万源祠），位于福建省龙岩市上杭县古田镇溪背村。始建于清道光二十八年（1848年），是一座典型的客家宗祠建筑。会址背靠参天古木的社下山，面朝视野开阔的田野，白墙青瓦，庄重古朴。大门横匾上“古田会议永放光芒”八个红色大字，在绿树掩映下熠熠生辉。",
            "history": "1929年12月28日至29日，中国共产党红军第四军第九次代表大会在此隆重召开。毛泽东同志主持会议并作了重要报告。会议一致通过了毛泽东起草的《古田会议决议》。这次会议解决了如何把以农民为主要成分的军队建设成为无产阶级性质的新型人民军队这个根本性问题。",
            "spirit": "古田会议确立了<b>“思想建党、政治建军”</b>的原则。它是人民军队建设史上的重要里程碑，军魂在此铸就。"
        },
        "nanchang": {
            "title": "南昌八一起义纪念馆",
            "tags": ["江西南昌", "军旗升起", "第一枪", "中西合璧建筑"],
            "desc": "南昌八一起义纪念馆位于江西省南昌市中山路，其旧址原为“江西大旅社”。这座建成于1925年的灰色五层砖瓦楼房，是当时南昌城的最高建筑，外观呈现出鲜明的中西合璧风格。宏伟的建筑外观与内部复原的统帅部场景，无声地诉说着那段惊心动魄的历史。",
            "history": "1927年8月1日凌晨，在周恩来、贺龙、叶挺、朱德、刘伯承等人的领导下，两万余名起义军在此集结，打响了武装反抗国民党反动派的第一枪。这次起义宣告了中国共产党把中国革命进行到底的坚定立场，标志着中国共产党独立领导革命战争、创建人民军队和武装夺取政权的开始。",
            "spirit": "<b>“八一”</b>二字从此成为中国人民解放军的醒目徽标，南昌也因此被誉为“军旗升起的地方”和“英雄城”。"
        },
        "jinggangshan": {
            "title": "井冈山革命博物馆",
            "tags": ["江西井冈山", "革命摇篮", "农村包围城市", "天下第一山"],
            "desc": "井冈山被誉为“中国革命的摇篮”。井冈山革命博物馆依山而建，气势恢宏，顶层设计采用客家民居通透式风格。馆内珍藏了大量珍贵的革命文物，如毛泽东当年用过的油灯、朱德挑粮的扁担等，全景式地展示了井冈山斗争时期的峥嵘岁月。",
            "history": "1927年10月，毛泽东率领秋收起义部队到达井冈山，创立了中国第一个农村革命根据地。在这里，中国共产党人点燃了“工农武装割据”的星星之火，开辟了<b>“农村包围城市、武装夺取政权”</b>的中国革命特色道路。著名的黄洋界保卫战就发生于此。",
            "spirit": "<b>井冈山精神</b>：坚定信念、艰苦奋斗，实事求是、敢闯新路，依靠群众、勇于胜利。"
        },
        "zunyi": {
            "title": "遵义会议会址",
            "tags": ["贵州遵义", "伟大转折", "1935年", "中西合璧"],
            "desc": "遵义会议会址位于贵州省遵义市老城子尹路96号，原为黔军将领柏辉章的私邸。建筑为二层砖木结构，融合了中式回廊与西式窗花风格，红墙青瓦，曲径回廊。这是当年红军长征途中占领的唯一一座城市里最好的建筑，见证了中国革命史上最惊险的一幕。",
            "history": "1935年1月15日至17日，中共中央政治局在此召开扩大会议。会议集中全力解决了当时具有决定意义的军事和组织问题，结束了“左”倾教条主义错误在中央的统治，确立了毛泽东在党和红军中的领导地位。",
            "spirit": "遵义会议在极端危急的历史关头，<b>挽救了党，挽救了红军，挽救了中国革命</b>，是党的历史上一个生死攸关的转折点。"
        },
        "luding": {
            "title": "泸定桥",
            "tags": ["四川甘孜", "大渡河", "铁索桥", "22勇士"],
            "desc": "泸定桥始建于清康熙四十四年（1705年），横跨水流湍急的大渡河，是连接藏汉交通的咽喉要道。桥身由13根碗口粗的铁链组成，其中9根作底链，4根分两侧作扶手，全长103米。桥下是波涛汹涌的河水，地势可谓“一夫当关，万夫莫开”。",
            "history": "1935年5月29日，中央红军红四团的22名勇士，冒着敌人的密集火力，攀踏着悬空的铁索，冲过大渡河，夺取了泸定桥。这一奇迹粉碎了蒋介石企图让红军成为“石达开第二”的梦想。",
            "spirit": "<b>飞夺泸定桥</b>是长征中最为惊心动魄的战役之一，展现了红军战士不畏艰险、勇往直前的英雄气概。"
        },
        "monument": {
            "title": "人民英雄纪念碑",
            "tags": ["北京天安门", "国家象征", "永垂不朽", "第一碑"],
            "desc": "人民英雄纪念碑矗立于北京天安门广场中心，是新中国为了纪念在人民解放战争和人民革命中牺牲的人民英雄而建立的。碑通高37.94米，碑身正面镌刻着毛泽东题写的“人民英雄永垂不朽”八个镏金大字，背面是周恩来题写的碑文。",
            "history": "1949年9月30日奠基，1958年建成。碑座下层四面镶嵌着十幅巨大的汉白玉浮雕，包括“虎门销烟”、“金田起义”、“武昌起义”、“五四运动”、“五卅运动”、“南昌起义”、“抗日游击战”、“胜利渡长江”等。",
            "spirit": "它概括了从1840年鸦片战争到1949年解放战争的一百多年间，中国人民反帝反封建的革命斗争史，是<b>中华民族精神的丰碑</b>。"
        },
        "ccp": {
            "title": "中共一大会址",
            "tags": ["上海兴业路", "党的诞生地", "石库门", "1921年"],
            "desc": "位于上海市兴业路76号（原望志路106号），是一幢典型的上海石库门里弄建筑。青红砖相间的清水外墙，黑漆大门，门楣上部有拱形堆塑花饰。在这座看似普通的民居中，诞生了改变中国历史走向的政党。",
            "history": "1921年7月23日，中国共产党第一次全国代表大会在此秘密召开。大会通过了中国共产党第一个纲领和决议，选举产生了中央局。毛泽东曾说：“中国产生了共产党，这是开天辟地的大事变。”",
            "spirit": "这里是中国共产党的<b>“产房”</b>，也是中国共产党人的精神家园。红色的起点，由此延伸。"
        },
        "tiananmen": {
            "title": "天安门广场",
            "tags": ["北京", "世界最大广场", "开国大典", "国家心脏"],
            "desc": "天安门广场位于北京市中心，南北长880米，东西宽500米，面积达44万平方米，是世界上最大的城市中心广场。广场北端是雄伟的天安门城楼，中央矗立着人民英雄纪念碑，南端是毛主席纪念堂，东侧是国家博物馆，西侧是人民大会堂。",
            "history": "天安门原名承天门，始建于明永乐十五年。1949年10月1日，毛泽东在天安门城楼上庄严宣告中华人民共和国成立。这里见证了五四运动、一二·九运动等重大历史事件，也见证了新中国历次盛大阅兵。",
            "spirit": "它是<b>新中国的象征</b>，记载了中国人民不屈不挠的革命精神和大无畏的英雄气概。"
        },
        "xibaipo": {
            "title": "西柏坡纪念馆",
            "tags": ["河北平山", "最后指挥所", "两个务必", "三大战役"],
            "desc": "西柏坡位于河北省平山县，是一个依山傍水的小村庄。纪念馆内复原了当年的中央军委作战室、七届二中全会会址等重要场景。那个“磨盘上布下百万兵”的传奇指挥所，至今仍保留着朴素的风貌。",
            "history": "1948年5月至1949年3月，中共中央在此办公。在这里，党中央指挥了震惊中外的<b>辽沈、淮海、平津三大战役</b>，决定了中国的命运；召开了党的七届二中全会，毛泽东提出了著名的“两个务必”。",
            "spirit": "周恩来评价：“西柏坡是毛主席和党中央进入北平，解放全中国的<b>最后一个农村指挥所</b>。”"
        },
        "baotashan": {
            "title": "延安宝塔山",
            "tags": ["陕西延安", "精神灯塔", "岭山寺塔", "革命圣地"],
            "desc": "宝塔山古称嘉岭山，位于延安城东南，延河之滨。山顶的唐代宝塔（岭山寺塔）高44米，楼阁式砖塔结构，是历史名城延安的标志性建筑。登塔远眺，革命圣地全貌尽收眼底。",
            "history": "“几回回梦里回延安，双手搂定宝塔山。”在革命战争年代，宝塔山是中国革命圣地的象征，是指引无数有志青年奔向光明的灯塔。中共中央在延安十三年，领导了抗日战争和解放战争。",
            "spirit": "它象征着<b>延安精神</b>：自力更生、艰苦奋斗的创业精神，全心全意为人民服务的精神。"
        },
        "nanhu": {
            "title": "嘉兴南湖红船",
            "tags": ["浙江嘉兴", "红船精神", "一大闭幕", "启航"],
            "desc": "停泊在浙江嘉兴南湖畔的一艘单夹弄丝网船（画舫）。1921年8月初，因上海法租界巡捕侵扰，中共一大代表们转移至此，在雨中泛舟，继续举行会议。这艘普通的画舫，因此承载了特殊的历史重量。",
            "history": "在这艘船上，中共一大胜利闭幕，正式宣告了中国共产党的诞生。会议通过了党的第一个纲领和第一个决议，选举了党的中央领导机构。",
            "spirit": "<b>红船精神</b>：开天辟地、敢为人先的首创精神，坚定理想、百折不挠的奋斗精神，立党为公、忠诚为民的奉献精神。"
        },
        "ruijin": {
            "title": "瑞金革命遗址",
            "tags": ["江西瑞金", "红色故都", "共和国摇篮", "红井"],
            "desc": "瑞金叶坪、沙洲坝等地保存了大量革命旧址。著名的“红井”就在这里，井旁立碑刻有“吃水不忘挖井人，时刻想念毛主席”。这里的建筑多为赣南客家风格，简陋的土墙屋见证了国家政权的雏形。",
            "history": "1931年11月7日，中华苏维埃共和国临时中央政府在瑞金宣告成立，毛泽东当选为中央执行委员会主席。这是中国共产党建立国家政权的首次尝试。",
            "spirit": "瑞金被誉为<b>“共和国摇篮”</b>，为后来新中国的建立积累了宝贵的治国理政经验。"
        },
        "shanghai_flower": {
            "title": "上海建党百年花坛",
            "tags": ["上海", "百年庆典", "城市景观", "繁荣昌盛"],
            "desc": "这是上海市为庆祝中国共产党成立100周年而特别设置的大型立体花坛装置（如位于人民广场、外滩或中共一大会址附近）。花坛通常以红黄为主色调，融合了石库门、红船、白玉兰等上海地标元素与红色文化符号。",
            "history": "2021年，上海作为党的诞生地，全城布置了大量精美的花卉景观。这些花坛不仅美化了城市，更营造了隆重热烈的节日氛围，吸引了无数市民游客打卡留念。",
            "spirit": "花团锦簇象征着<b>祖国的繁荣昌盛</b>和人民生活的幸福美好，也是对百年奋斗历程的深情礼赞。"
        }
    }

    # 模糊匹配逻辑
    for key in db:
        if key in name or name in key:
            return db[key]
            
    # 特殊处理：如果没匹配到，尝试用别名
    if "first_congress" in name: return db["ccp"]
    if "yanan" in name: return db["baotashan"]
    if "ship" in name: return db["nanhu"]
    if "flower" in name: return db["shanghai_flower"]
    if "jiangxi" in name: return db["ruijin"]

    return default_info

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

# --- 5. 密码锁 (Sidebar) ---
ADMIN_PASSWORD = "123"
with st.sidebar:
    st.markdown("### ⚙️ Admin")
    password = st.text_input("Password", type="password")
    st.caption("© 2025 一拍即知")

if password != ADMIN_PASSWORD:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #86868b;">
        <h2>🔒 Locked</h2>
        <p>Please enter the password in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 6. 主界面构建 ---

# Header
st.markdown("""
<div class="header-container">
    <div class="app-title">一拍即知</div>
    <div class="app-subtitle">
        <span style="margin: 0 10px; color: #d2d2d7;">|</span> 让每一张照片，都有故事。
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("⚠️ 系统提示：模型文件未找到，请检查服务器配置。")
    st.stop()

# 布局
col_left, col_right = st.columns([3, 4], gap="large")

with col_left:
    st.markdown("##### 📤 上传影像 / Upload")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # 图片容器样式
        st.markdown("""
        <style>
            .uploaded-img { 
                border-radius: 12px; 
                box-shadow: 0 8px 24px rgba(0,0,0,0.1); 
                margin-top: 20px;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    else:
        st.info("👈 请点击上方区域选择照片")

# 推理与展示
if uploaded_file is not None:
    # 进度条
    progress_bar = col_left.progress(0)
    for i in range(100):
        time.sleep(0.002)
        progress_bar.progress(i + 1)
    progress_bar.empty()
    
    # AI 推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    class_id = predicted.item()
    raw_name = class_names[class_id]
    score = confidence.item() * 100
    
    # 获取详细资料
    info = get_landmark_info(raw_name)

    # --- 右侧信息展示 (HTML 渲染修复版) ---
    with col_right:
        # 使用 textwrap.dedent 清除缩进，防止被识别为代码块
        html_content = textwrap.dedent(f"""
            <div class="info-card">
                <div class="landmark-header">
                    <div class="landmark-title">{info['title']}</div>
                    <div class="tag-container">
                        {''.join([f'<span class="tag-pill">{tag}</span>' for tag in info['tags']])}
                        <span class="match-score">AI Match {score:.1f}%</span>
                    </div>
                </div>
                
                <div class="section-title">🏛️ 地标简介</div>
                <div class="section-content">{info['desc']}</div>
                
                <div class="section-title">📜 历史文脉</div>
                <div class="section-content">{info['history']}</div>
                
                <div class="section-title">🔥 精神内涵</div>
                <div class="section-content">{info['spirit']}</div>
            </div>
        """)
        
        st.markdown(html_content, unsafe_allow_html=True)
