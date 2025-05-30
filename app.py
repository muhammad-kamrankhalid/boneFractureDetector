import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

# Monkey patch for PyTorch >=2.6
_real_torch_load = torch.load
def custom_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = custom_load

# Set page config
st.set_page_config(page_title="Bone Fracture Detection", layout="centered")

# Inject custom CSS for background image
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://badgut.org/wp-content/uploads/Image-Content-PPI-Bone.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App UI
st.title("🦴 Bone Fracture Detection in X-ray Images")
st.write("Upload an X-ray image and detect bone fractures using a trained YOLOv8 model.")

# Load the model
@st.cache_resource
def load_model():
    return YOLO("boneBest.pt")  # Make sure this model is placed in the same folder

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Run YOLO detection
    results = model(image)
    annotated_img = results[0].plot()

    st.image(annotated_img, caption="Detected Fractures", use_column_width=True)

    # List detected boxes
    st.write("### Detection Details:")
    boxes = results[0].boxes
    if boxes and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf = box.conf.item()
            st.write(f"- `{cls_name}` with confidence **{conf:.2f}**")
    else:
        st.success("✅ No fractures detected.")
