import streamlit as sl
from PIL import Image

sl.title("♻️🗑️ Waste Classifier by The Decompilers 🗑️♻️")
sl.write("Confused about what kind of waste this is? Upload a photo! We'll tell you if it's garbage, compost, or recycling!")

uploaded_file = sl.file_uploader(" ",type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    sl.image(image, use_container_width=True)

    sl.markdown("Our prediction is: ")
    sl.info("_(working on model)_")