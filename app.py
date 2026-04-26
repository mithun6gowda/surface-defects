import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.model import get_model

# Load class names (same order as training)
class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Load model
model = get_model(num_classes=6)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("🔍 Surface Defect Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    st.success(f"Predicted Defect: {class_names[pred.item()]}")