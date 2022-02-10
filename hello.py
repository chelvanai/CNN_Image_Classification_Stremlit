import json

import streamlit as st
import torch
from PIL import Image

from model import Net
import test

st.title("Upload + Classification Example")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save("test.jpg")

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    model = Net().to(device)

    with open('data.json', 'r') as f:
        labels = json.load(f)

    res = test.test("test.jpg", model, labels)
    st.write(res)

