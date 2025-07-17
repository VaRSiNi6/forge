# ==============================
# ✅ STEP 1: Import everything
# ==============================

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import streamlit as st
import requests
import json
import io

# ==============================
# ✅ STEP 2: Setup device
# ==============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================
# ✅ STEP 3: Load model
# ==============================

NUM_CLASSES = 4
class_names = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

@st.cache_resource
def load_model():
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load("resnet50_grapes.pt", map_location=device))  # Upload this file to root
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# ==============================
# ✅ STEP 4: Image transform
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# ✅ Streamlit UI
# ==============================

st.title("🍇 Grape Leaf Disease Predictor")
st.markdown("Upload a grape leaf image to detect disease and get treatment advice.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # ==============================
    # ✅ Prediction
    # ==============================

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]

    st.success(f"✅ **Predicted Disease:** {predicted_label}")

    # ==============================
    # ✅ Static Advice
    # ==============================

    disease_info = {
        "Black Rot": {
            "cause": "Black Rot is caused by the fungus *Guignardia bidwellii*. It spreads in warm, humid conditions.",
            "remedy": "Prune infected leaves, apply fungicides like mancozeb. Maintain vineyard hygiene.",
            "fertilizer": "Balanced NPK, avoid excess nitrogen.",
            "tips": "Ensure good air flow, monitor during warm wet seasons."
        },
        "ESCA": {
            "cause": "ESCA is caused by multiple wood-rotting fungi. Leads to leaf striping and dieback.",
            "remedy": "Remove infected wood, disinfect pruning tools.",
            "fertilizer": "Keep soil nutrients balanced.",
            "tips": "Avoid pruning wounds, apply protectants if possible."
        },
        "Healthy": {
            "cause": "No visible disease detected.",
            "remedy": "Continue good care practices.",
            "fertilizer": "Balanced feeding per soil tests.",
            "tips": "Monitor regularly for early signs."
        },
        "Leaf Blight": {
            "cause": "Leaf Blight is often due to fungi like *Pseudocercospora vitis*.",
            "remedy": "Remove infected leaves, use copper or chlorothalonil sprays.",
            "fertilizer": "Maintain soil health, avoid excess nitrogen.",
            "tips": "Prune for good air flow, manage humidity."
        }
    }

    info = disease_info[predicted_label]

    st.subheader("📋 Static Advice")
    st.markdown(f"- **Cause:** {info['cause']}")
    st.markdown(f"- **Remedy:** {info['remedy']}")
    st.markdown(f"- **Fertilizer:** {info['fertilizer']}")
    st.markdown(f"- **Tips:** {info['tips']}")

    # ==============================
    # ✅ DeepSeek API Call
    # ==============================

    st.subheader("🤖 AI-Powered Summary")

    OPENROUTER_API_KEY = "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"  # Replace with your key

    if OPENROUTER_API_KEY.startswith("sk-"):
        query = f"""
        My grape plant has {predicted_label}.
        Give me a SHORT, practical answer for farmers:
        - Main cause (1–2 lines)
        - Quick fix or treatment
        - Which fungicide or fertilizer to use
        - 2–3 farmer tips.
        Keep it under 100 words.
        """

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "deepseek/deepseek-chat:free",
            "messages": [{"role": "user", "content": query}]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            st.markdown(answer)
        else:
            st.warning("DeepSeek API call failed.")
    else:
        st.info("Add your `OPENROUTER_API_KEY` in the code to get AI-powered response.")
