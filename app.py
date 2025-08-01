import streamlit as sl
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@sl.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("model_1.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

sl.title("♻️🗑️ Waste Classifier by The Decompilers 🗑️♻️")
sl.write("Confused about what kind of waste this is? Upload a photo! We'll tell you if it's garbage, compost, or recycle!")

uploaded_file = sl.file_uploader(" ",type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    sl.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
        _, pred = torch.max(output, 1)
    
    class_names = ['Compost', 'Garbage', 'Recycle']
    confidence = probabilities[pred.item()] * 100
    
    sl.markdown("### Our prediction is:")
    sl.success(f"**{class_names[pred.item()]}** (Confidence: {confidence:.2f}%)")