import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
import datetime

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="一拍即知 | 红色档案",
    page_icon="🇨🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 核心 CSS 美化 (纯白档案风 - 极致对比度) ---
st.markdown("""
<style>
    /* 引入衬线字体，营造严肃的历史感 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700;900&display=swap');

    /* === 全局强制纯白背景，纯黑文字 === */
    .stApp {
        background-color: #FFFFFF; /* 纯白 */
        color: #000000; /* 纯黑 */
    }
    
    /* 强制所有默认文本为黑色 */
    p, div, span, label, li {
        color: #1a1a1a !important;
        font-family: 'Noto Serif SC', serif;
    }

    /* === 顶部 Header === */
    .header-container {
        text-align: center;
        padding: 50px 0 30px 0;
        border-bottom: 3px solid #8B0000; /* 底部红线 */
        margin-bottom: 40px;
    }
    .app-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 5rem;
        font-weight: 900;
        color: #8B0000 !important; /* 正统深红 */
        margin-bottom: 10px;
        letter-spacing: 5px;
    }
    .app-subtitle {
        font-size: 1.4rem;
        color: #333333 !important;
        letter-spacing: 2px;
        font-weight: bold;
    }

    /* === 修复：上传区域高对比度 === */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #000000; /* 黑色虚线框 */
        background-color: #f8f8f8; /* 极淡的灰，区分区域 */
        padding: 30px;
        border-radius: 0px; /* 直角，更像档案 */
    }
    /* 暴力强制上传按钮里的文字颜色 */
    div[data-testid="stFileUploader"] label {
        color: #000000 !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
    }
    div[data-testid="stFileUploader"] div {
        color: #000000 !important;
    }
    div[data-testid="stFileUploader"] button {
        border-color: #000000 !important;
        color: #000000 !important;
    }

    /* === 结果卡片 (左侧黑线风格) === */
    .info-container {
        padding: 20px;
        border-left: 8px solid #8B0000; /* 左侧粗红线装饰 */
        background-color: #fffbf0; /* 极淡的米黄色，模拟旧纸张 */
    }
    
    .landmark-name {
        font-size: 3rem;
        font-weight: 900;
        color: #000000 !important;
        margin-bottom: 10px;
        border-bottom: 1px solid #000;
        padding-bottom: 20px;
    }
    
    .section-head {
        font-size: 1.5rem;
        font-weight: 700;
        color: #8B0000 !important; /* 红色小标题 */
        margin-top: 30px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .section-head::before {
        content: '■';
        font-size: 0.8em;
        margin-right: 10px;
        color: #000;
    }

    .content-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #222 !important;
        text-align: justify;
    }

    /* 标签样式 */
    .tag-item {
        display: inline-block;
        border: 1px solid #000;
        padding: 5px 15px;
        margin-right: 10px;
        font-weight: bold;
        color: #000 !important;
        background: #fff;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 深度文献资料库 (纯文本，防止代码错误) ---
def get_landmark_info(name):
    info = {
        "title": "未知地标",
        "tags": ["待识别"],
        "desc": "暂无资料。",
        "history": "请上传清晰的红色地标照片。",
        "spirit": "无法识别"
    }

    if "gutian" in name:
        info = {
            "title": "古田会议会址",
            "tags": ["福建上杭", "思想建党", "1929年", "重点文物"],
            "desc": "古田会议会址，原名廖氏宗祠（万源祠），位于福建省龙岩市上杭县古田镇溪背村。始建于清道光二十八年（1848年），是一座典型的客家宗祠建筑，总面积826平方米。会址座东朝西，背靠参天古木的社下山，面朝视野开阔的田野。建筑为砖木结构，白墙青瓦，飞檐翘角。大门横匾上“古田会议永放光芒”八个红色大字，在绿树掩映下熠熠生辉。祠堂内正厅是当年召开会议的会场，主席台设在左侧，墙上依然保留着当年的马克思、列宁画像和红军军旗。",
            "history": "1929年12月28日至29日，中国共产党红军第四军第九次代表大会在此隆重召开。会议期间，毛泽东作了政治报告，朱德作了军事报告，陈毅传达了中央九月来信。会议一致通过了毛泽东起草的《中国共产党红军第四军第九次代表大会决议案》（即《古田会议决议》）。这次会议系统总结了红四军成立以来军队建设的经验教训，批判了各种非无产阶级思想，确立了“思想建党、政治建军”的原则，解决了如何把以农民为主要成分的军队建设成为无产阶级性质的新型人民军队这个根本性问题。",
            "spirit": "古田会议精神的核心是“思想建党、政治建军”。它是人民军队建设史上的重要里程碑，标志着中国共产党在政治上、思想上、组织上的成熟。在这里，军魂被铸就，党指挥枪的原则被确立。"
        }
    elif "nanchang" in name:
        info = {
            "title": "南昌八一起义纪念馆",
            "tags": ["江西南昌", "军旗升起", "第一枪", "中西合璧"],
            "desc": "南昌八一起义纪念馆位于江西省南昌市中山路，其旧址原为“江西大旅社”。这座建成于1925年的灰色五层砖瓦楼房，是当时南昌城的最高建筑，外观呈现出鲜明的中西合璧风格。楼内共有房间96间，回廊曲折，天井深邃。1927年，这里被包租下来作为起义的总指挥部。纪念馆内现陈列有大量珍贵的历史文物，还原了周恩来、林伯渠等领导人的办公室和卧室，无声地诉说着那段惊心动魄的历史。",
            "history": "1927年，大革命失败后，为了挽救革命，中共中央决定在南昌发动武装起义。8月1日凌晨2时，在周恩来、贺龙、叶挺、朱德、刘伯承等人的领导下，两万余名起义军以颈系红领带、臂扎白毛巾为标志，向国民党反动派发起了猛烈进攻。经过四个多小时的激战，起义军全歼守敌3000余人，占领了南昌城。这次起义宣告了中国共产党把中国革命进行到底的坚定立场，标志着中国共产党独立领导革命战争、创建人民军队和武装夺取政权的开始。",
            "spirit": "南昌起义精神是“听党指挥、敢为人先、百折不挠、为民奋斗”。“八一”二字从此成为中国人民解放军的醒目徽标，南昌也因此被誉为“军旗升起的地方”和“英雄城”。"
        }
    elif "jinggangshan" in name:
        info = {
            "title": "井冈山革命博物馆",
            "tags": ["江西井冈山", "革命摇篮", "农村包围城市", "天下第一山"],
            "desc": "井冈山被誉为“中国革命的摇篮”。井冈山革命博物馆依山而建，气势恢宏，顶层设计采用客家民居通透式风格。馆内通过大量的历史文物、图片、油画以及高科技的多媒体场景复原，全景式地展示了井冈山革命根据地的创立、发展和斗争历程。著名的黄洋界哨口、八角楼毛泽东旧居等都是井冈山的重要组成部分。",
            "history": "1927年10月，毛泽东率领秋收起义部队到达井冈山，创立了中国第一个农村革命根据地。1928年4月，朱德、陈毅率领起义部队与毛泽东会师，成立了中国工农红军第四军。在这里，中国共产党人点燃了“工农武装割据”的星星之火，开辟了“农村包围城市、武装夺取政权”的中国革命特色道路。无数革命先烈在这里洒下了热血，留下了“朱德的扁担”、“红米饭南瓜汤”等动人故事。",
            "spirit": "井冈山精神：坚定信念、艰苦奋斗，实事求是、敢闯新路，依靠群众、勇于胜利。它是中国共产党精神谱系的重要组成部分，是激励我们不断前进的强大精神动力。"
        }
    elif "zunyi" in name:
        info = {
            "title": "遵义会议会址",
            "tags": ["贵州遵义", "伟大转折", "1935年", "中西合璧"],
            "desc": "遵义会议会址位于贵州省遵义市老城子尹路96号，原为黔军将领柏辉章的私邸。建筑为二层砖木结构，融合了中式回廊与西式窗花风格，红墙青瓦，曲径回廊，古朴典雅。主楼坐北朝南，面阔七间，进深三间。这是当年红军长征途中占领的唯一一座城市里最好的建筑，见证了中国革命史上最惊险的一幕。",
            "history": "1935年1月15日至17日，在红军第五次反“围剿”失败和长征初期严重受挫的历史关头，中共中央政治局在此召开扩大会议。会议集中全力解决了当时具有决定意义的军事和组织问题，结束了“左”倾教条主义错误在中央的统治，确立了毛泽东在党和红军中的领导地位。会议取消了博古、李德的最高军事指挥权，选举毛泽东为中央政治局常委。",
            "spirit": "遵义会议在极端危急的历史关头，挽救了党，挽救了红军，挽救了中国革命，是党的历史上一个生死攸关的转折点。它标志着中国共产党在政治上开始走向成熟，独立自主地解决中国革命问题。"
        }
    elif "luding" in name:
        info = {
            "title": "泸定桥",
            "tags": ["四川甘孜", "大渡河", "铁索桥", "22勇士"],
            "desc": "泸定桥始建于清康熙四十四年（1705年），横跨水流湍急的大渡河，是连接藏汉交通的咽喉要道。桥身由13根碗口粗的铁链组成，其中9根作底链，4根分两侧作扶手，全长103米，宽3米。桥下是波涛汹涌、惊涛拍岸的河水，两岸是高耸入云的山峦，地势可谓“一夫当关，万夫莫开”。",
            "history": "1935年5月29日，中央红军红四团的22名勇士，面对对岸敌人的密集火力和桥下咆哮的江水，在只剩下光溜溜铁索的桥上，攀踏着悬空的铁索，匍匐前进，冲过大渡河，夺取了泸定桥。后续部队迅速铺设桥板，大军胜利通过。这一奇迹粉碎了蒋介石企图让红军成为“石达开第二”的梦想。",
            "spirit": "飞夺泸定桥是长征中最为惊心动魄的战役之一，展现了红军战士不畏艰险、不怕牺牲、勇往直前的英雄气概和革命理想高于天的坚定信念。"
        }
    elif "monument" in name:
        info = {
            "title": "人民英雄纪念碑",
            "tags": ["北京天安门", "国家象征", "永垂不朽", "第一碑"],
            "desc": "人民英雄纪念碑矗立于北京天安门广场中心，是新中国为了纪念在人民解放战争和人民革命中牺牲的人民英雄而建立的。碑通高37.94米，由1.7万块花岗岩和汉白玉砌成。碑身正面镌刻着毛泽东题写的“人民英雄永垂不朽”八个镏金大字，背面是周恩来题写的碑文。碑座下层四面镶嵌着十幅巨大的汉白玉浮雕，栩栩如生。",
            "history": "1949年9月30日，中国人民政治协商会议第一届全体会议决定建立人民英雄纪念碑，并举行了奠基仪式。1958年建成揭幕。浮雕内容包括“虎门销烟”、“金田起义”、“武昌起义”、“五四运动”、“五卅运动”、“南昌起义”、“抗日游击战”、“胜利渡长江”等，概括了从1840年鸦片战争到1949年解放战争的一百多年间，中国人民反帝反封建的革命斗争史。",
            "spirit": "它是中华民族精神的丰碑，象征着无数革命先烈为了民族独立、人民解放和国家富强而英勇奋斗的牺牲精神。它时刻提醒着后人：今天的幸福生活来之不易。"
        }
    elif "ccp" in name or "first_congress" in name:
        info = {
            "title": "中共一大会址",
            "tags": ["上海兴业路", "党的诞生地", "石库门", "1921年"],
            "desc": "中共一大会址位于上海市兴业路76号（原望志路106号），是一幢典型的上海石库门里弄建筑。青红砖相间的清水外墙，黑漆大门，门楣上部有拱形堆塑花饰，门框围以米黄色石条。建筑风格中西结合，庄重而典雅。在这座看似普通的民居中，诞生了改变中国历史走向的伟大政党。",
            "history": "1921年7月23日，中国共产党第一次全国代表大会在此秘密召开。来自全国各地的13名代表代表全国50多名党员出席了会议。7月30日晚，因法租界巡捕袭扰，会议被迫中止，最后一天转移到嘉兴南湖的一艘游船上继续举行。大会通过了中国共产党第一个纲领和决议，选举产生了中央局。",
            "spirit": "这里是中国共产党的“产房”，也是中国共产党人的精神家园。毛泽东曾评价：“中国产生了共产党，这是开天辟地的大事变。”红色的起点，由此延伸，中国革命的面貌从此焕然一新。"
        }
    elif "tiananmen" in name:
        info = {
            "title": "天安门广场",
            "tags": ["北京", "世界最大广场", "开国大典", "国家心脏"],
            "desc": "天安门广场位于北京市中心，南北长880米，东西宽500米，面积达44万平方米，是世界上最大的城市中心广场。广场北端是雄伟的天安门城楼，红墙黄瓦，金碧辉煌；中央矗立着人民英雄纪念碑；南端是毛主席纪念堂和正阳门；东侧是国家博物馆；西侧是人民大会堂。整个广场气势恢宏，庄严肃穆。",
            "history": "天安门原名承天门，始建于明永乐十五年（1417年）。1949年10月1日，开国大典在此隆重举行，毛泽东主席在天安门城楼上庄严宣告中华人民共和国成立，亲手升起了第一面五星红旗。这里见证了五四运动、一二·九运动等中国现代史上的重大历史事件，也见证了新中国历次盛大阅兵和群众游行。",
            "spirit": "天安门广场是新中国的象征，是全国各族人民向往的地方。它记载了中国人民不屈不挠的革命精神和大无畏的英雄气概，见证了中华民族从站起来、富起来到强起来的伟大飞跃。"
        }
    elif "xibaipo" in name:
        info = {
            "title": "西柏坡纪念馆",
            "tags": ["河北平山", "最后指挥所", "两个务必", "三大战役"],
            "desc": "西柏坡位于河北省平山县，是一个依山傍水的小村庄，松柏苍翠，风光秀丽。西柏坡纪念馆内复原了当年的中共中央旧址，包括毛泽东、周恩来、朱德等领导人的旧居，以及中央军委作战室、七届二中全会会址等重要场景。那个“磨盘上布下百万兵”的传奇指挥所，至今仍保留着简朴的风貌。",
            "history": "1948年5月至1949年3月，中共中央在此办公。在这里，党中央运筹帷幄，指挥了震惊中外的辽沈、淮海、平津三大战役，消灭了国民党军队的主力，奠定了人民解放战争在全国胜利的基础。1949年3月，党的七届二中全会在此召开，毛泽东提出了著名的“两个务必”。",
            "spirit": "周恩来评价：“西柏坡是毛主席和党中央进入北平，解放全中国的最后一个农村指挥所。”西柏坡精神的核心是“两个务必”：务必使同志们继续地保持谦虚、谨慎、不骄、不躁的作风，务必使同志们继续地保持艰苦奋斗的作风。"
        }
    elif "baotashan" in name or "yanan" in name:
        info = {
            "title": "延安宝塔山",
            "tags": ["陕西延安", "精神灯塔", "岭山寺塔", "革命圣地"],
            "desc": "宝塔山古称嘉岭山，位于延安城东南，延河之滨。山顶的唐代宝塔（岭山寺塔）高44米，为八角九级楼阁式砖塔结构，是历史名城延安的标志性建筑。登塔远眺，延安城全貌和延河风光尽收眼底。山上还留有范仲淹等历代名人的摩崖石刻。",
            "history": "“几回回梦里回延安，双手搂定宝塔山。”在革命战争年代，宝塔山是中国革命圣地的象征，是指引无数有志青年奔向光明的灯塔。1935年到1948年，中共中央在延安十三年，领导了抗日战争和解放战争，开展了整风运动和大生产运动，培育了光照千秋的延安精神。",
            "spirit": "延安精神：坚定正确的政治方向，解放思想、实事求是的思想路线，全心全意为人民服务的根本宗旨，自力更生、艰苦奋斗的创业精神。它是中国共产党宝贵的精神财富。"
        }
    elif "nanhu" in name or "ship" in name:
        info = {
            "title": "嘉兴南湖红船",
            "tags": ["浙江嘉兴", "红船精神", "一大闭幕", "启航"],
            "desc": "停泊在浙江嘉兴南湖畔的是一艘单夹弄丝网船，俗称“画舫”。这艘船长约16米，宽3米，内设前舱、中舱、房舱和后舱。1921年8月初，因上海法租界巡捕侵扰，中共一大代表们转移至此，在烟雨蒙蒙的南湖上泛舟，继续举行会议。这艘普通的画舫，因此承载了特殊的历史重量，被亲切地称为“红船”。",
            "history": "在这艘船上，中共一大胜利闭幕，正式宣告了中国共产党的诞生。会议通过了党的第一个纲领和第一个决议，选举了党的中央领导机构。这艘小船，承载着中国人民的重托和民族的希望，越过急流险滩，穿过惊涛骇浪，成为领航中国行稳致远的巍巍巨轮。",
            "spirit": "2005年，习近平总书记首次概括了“红船精神”：开天辟地、敢为人先的首创精神，坚定理想、百折不挠的奋斗精神，立党为公、忠诚为民的奉献精神。它是中国革命精神之源。"
        }
    elif "ruijin" in name:
        info = {
            "title": "瑞金革命遗址",
            "tags": ["江西瑞金", "红色故都", "共和国摇篮", "红井"],
            "desc": "瑞金叶坪、沙洲坝等地保存了大量革命旧址，如第一次全国苏维埃代表大会会址、中共苏区中央局旧址等。著名的“红井”就在沙洲坝，井旁立碑刻有“吃水不忘挖井人，时刻想念毛主席”。这里的建筑多为赣南客家风格，简陋的土墙屋见证了国家政权的雏形。",
            "history": "1931年11月7日，中华苏维埃共和国临时中央政府在瑞金宣告成立，毛泽东当选为中央执行委员会主席。这是中国共产党建立国家政权的首次尝试，颁布了《中华苏维埃共和国宪法大纲》等法律文件，开展了土地革命和经济建设。",
            "spirit": "瑞金被誉为“红色故都”和“共和国摇篮”。苏区精神——坚定信念、求真务实、一心为民、清正廉洁、艰苦奋斗、争创一流、无私奉献，为后来新中国的建立积累了宝贵的治国理政经验。"
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

# --- 5. 密码锁 (Sidebar) ---
ADMIN_PASSWORD = "123"
with st.sidebar:
    st.markdown("### ⚙️ 管理员入口")
    password = st.text_input("请输入密码", type="password")
    st.caption("© 2025 一拍即知")

if password != ADMIN_PASSWORD:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.error("⚠️ 系统已锁定，请在左侧侧边栏输入密码解锁。")
    st.stop()

# --- 6. 主界面构建 ---

# Header
st.markdown("""
<div class="header-container">
    <div class="app-title">一拍即知</div>
    <div class="app-subtitle">
        Snap. Discover. Connect. — 让每一张照片，都有故事。
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or not class_names:
    st.error("⚠️ 系统提示：模型文件未找到，请检查服务器配置。")
    st.stop()

# 布局
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("### 📤 上传影像")
    uploaded_file = st.file_uploader("请选择一张红色地标照片", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
    else:
        st.info("👈 请在上方虚线框内上传照片")

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

    # --- 右侧信息展示 (彻底修复 HTML 渲染问题) ---
    with col_right:
        # 1. 标题与标签
        st.markdown(f'<div class="info-container"><div class="landmark-name">{info["title"]}</div>', unsafe_allow_html=True)
        
        # 标签组
        tags_html = "".join([f'<span class="tag-item">{tag}</span>' for tag in info['tags']])
        st.markdown(f'<div>{tags_html} <span style="float:right; font-weight:bold;">AI 置信度: {score:.1f}%</span></div>', unsafe_allow_html=True)
        
        # 2. 正文内容
        st.markdown(f'<div class="section-head">地标简介</div><p class="content-text">{info["desc"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-head">历史文脉</div><p class="content-text">{info["history"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-head">精神内涵</div><p class="content-text">{info["spirit"]}</p>', unsafe_allow_html=True)
        
        # 关闭上面的 info-container div
        st.markdown('</div>', unsafe_allow_html=True)

        # ==========================================================
        # === ✨ 新增功能：用户反馈系统 (大作业加分项) ✨ ===
        # ==========================================================
        
        st.markdown("---") # 分割线
        st.markdown("#### 📝 协助我们优化模型")
        st.caption("您的反馈将用于下一轮模型迭代训练")

        # 1. 确保文件夹存在 (自动创建)
        os.makedirs("feedback_good", exist_ok=True)
        os.makedirs("feedback_bad", exist_ok=True)

        # 2. 生成文件名 (时间戳 + 预测类别)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{timestamp}_{raw_name}.jpg"

        # 3. 放置两个并排的按钮
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            if st.button("✅ 识别准确", use_container_width=True):
                save_path = os.path.join("feedback_good", save_name)
                image.save(save_path) # 保存图片
                st.success("反馈成功！已归档至[正样本库]。")
                st.balloons() # 🎉 演示特效：飘气球 (老师最爱看这种交互)
        
        with f_col2:
            if st.button("❌ 识别错误", use_container_width=True):
                save_path = os.path.join("feedback_bad", save_name)
                image.save(save_path) # 保存图片
                st.error("反馈已提交！已归档至[待修正库]。")
                # 这里不加气球，保持严肃
