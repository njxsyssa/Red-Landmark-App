import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. 设置页面
st.set_page_config(page_title="红色地标一拍即知", layout="centered")
st.title("📸 红色地标一拍即知")
st.write("上传一张红色景点的照片，模型将自动识别其背景信息。")

# 2. 加载类别名称
class_names = []
if os.path.exists("class_names.txt"):
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    st.error("未找到类别文件，请先运行 train.py 进行训练！")
    st.stop()

# 3. 加载模型
@st.cache_resource # 缓存模型，避免每次刷新都重新加载
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    try:
        model.load_state_dict(torch.load('red_landmark_model.pth', map_location=device))
    except FileNotFoundError:
        return None
        
    model = model.to(device)
    model.eval()
    return model

model = load_model()

if model is None:
    st.error("未找到模型文件 'red_landmark_model.pth'。请先运行 train.py。")
    st.stop()

# 4. 定义预处理 (必须与训练时验证集的处理一致)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. 图片上传与推理
uploaded_file = st.file_uploader("请选择一张图片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='上传的图片', use_column_width=True)
    
    st.write("正在识别中...")
    
    # 推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
    class_id = predicted.item()
    result_name = class_names[class_id]
    score = confidence[class_id].item()

    # 6. 结果展示
    st.success(f"识别结果：**{result_name}**")
    st.info(f"置信度：{score:.2f}%")

    # 根据PPT中的描述，这里可以后续添加具体的历史背景介绍 [cite: 7]
    # 这里做一个简单的示例映射
    st.markdown("---")
    st.subheader("📖 历史背景资料")
    if "gutian" in result_name:
        st.write("古田会议会址：1929年12月，中国工农红军第四军第九次党的代表大会在此召开...")
    elif "nanchang" in result_name:
        st.write("南昌八一起义纪念馆：1927年8月1日，中国共产党在这里打响了武装反抗国民党反动派的第一枪...")
    elif "jinggangshan" in result_name:
        st.write("井冈山革命博物馆：井冈山是中国革命的摇篮...")
    else:
        st.write(f"这是关于 {result_name} 的相关历史资料...")
