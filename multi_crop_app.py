import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
import requests
import json
import os

# Set device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Title
st.set_page_config(page_title="Crop Disease Detection", layout="centered")
st.title("🌿 Crop Disease Detection App")
st.markdown("Select your crop, upload or capture an image, and get disease prediction with remedies.")

# Crop selection
# Updated "NewCrop" to "Rice"
crop = st.selectbox("Choose your crop", ["Grapes", "Potato", "Peanut", "Tomato", "Rice"])

# Map crop to model path, labels, num_classes, and specific FC layer structure
model_info = {
    "Grapes": {
        "model_path": "resnet50_grapes.pt",
        "labels": ["Black Rot", "ESCA", "Healthy", "Leaf Blight"],
        "num_classes": 4,
        "fc_type": "single_linear", # Grapes uses a single nn.Linear layer for FC
        "remedies": {
            "Black Rot": "Use fungicides (e.g., Mancozeb, Myclobutanil). Prune and destroy infected plant parts. Improve air circulation.",
            "ESCA": "Prune out diseased wood. Disinfect tools. Manage irrigation to reduce stress. No chemical cure, focus on prevention and sanitation.",
            "Healthy": "Your grape plant appears healthy! Continue good cultural practices like proper pruning, irrigation, and nutrient management.",
            "Leaf Blight": "Remove affected leaves. Apply copper-based fungicides. Ensure good air circulation. Avoid overhead irrigation."
        }
    },
    "Potato": {
        "model_path": "resnet50_potato.pt", # Ensure this file exists in your project root
        "labels": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"],
        "num_classes": 3,
        "fc_type": "single_linear", # Potato uses a single nn.Linear layer for FC (confirmed)
        "remedies": {
            "Potato___Early_blight": "Apply fungicides containing chlorothalonil or azoxystrobin. Remove and destroy infected foliage. Practice crop rotation.",
            "Potato___Late_blight": "Use systemic fungicides (e.g., Ridomil Gold, Revus). Destroy infected plants. Ensure proper spacing for air circulation.",
            "Potato___healthy": "Your potato crop seems healthy! Maintain balanced fertilization, consistent watering, and monitor for early signs of disease."
        }
    },
    "Peanut": {
        "model_path": "resnet50_peanut.pt", # Ensure this file exists in your project root
        "labels": ["Peanut_rust", "Peanut_nutrition_deficiency", "Peanut_leaf_spot", "Peanut_healthy_leaf"],
        "num_classes": 4,
        "fc_type": "custom_sequential_peanut", # Peanut uses the specific custom sequential head (Linear 512, Dropout 0.3)
        "remedies": {
            "Peanut_rust": "Apply fungicides like tebuconazole or propiconazole. Plant resistant varieties. Rotate crops with non-host plants.",
            "Peanut_nutrition_deficiency": "Perform a soil test to identify specific deficiencies. Apply appropriate fertilizers (e.g., micronutrient mixes for iron/zinc deficiency).",
            "Peanut_leaf_spot": "Use resistant varieties and appropriate fungicides. Practice good field sanitation and crop rotation.",
            "Peanut_healthy_leaf": "The peanut leaf looks healthy! Continue good agronomic practices including proper soil testing and nutrient management."
        }
    },
    "Tomato": {
        "model_path": "resnet50_tomato.pt", # Based on your "tomato app.py" file
        "labels": ["Tomato__Tomato_mosaic_virus", "Tomato___Bacterial_spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato__blight"], # From label_map tomato.json
        "num_classes": 4, # Matches the labels for Tomato
        "fc_type": "single_linear", # Confirmed by your "tomato app.py"
        "remedies": { # From your "tomato app.py"
            "Tomato__Tomato_mosaic_virus": "Viral infection causing mottled, curled leaves and reduced yield. Mitigation: Remove infected plants, disinfect tools, use virus-free seeds. Fertilizer: Use phosphorus-rich fertilizer to support healthy growth.",
            "Tomato___Bacterial_spot": "Causes small, dark, water-soaked spots on leaves and fruits. Mitigation: Apply copper-based bactericide, remove infected plants. Fertilizer: Low nitrogen, high potassium fertilizers recommended.",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Leads to curled, yellowing leaves and stunted growth. Spread by whiteflies. Mitigation: Control whiteflies, remove infected plants, use resistant varieties. Fertilizer: Potassium-rich fertilizers help enhance immunity.",
            "Tomato__blight": "Fungal disease causing brown spots, yellowing, and leaf drop. Mitigation: Use fungicides, improve air circulation, avoid overhead watering. Fertilizer: Use balanced NPK fertilizer, add calcium for leaf strength."
        }
    },
    "Rice": { # Renamed from "NewCrop" to "Rice"
        "model_path": "resnet50_rice.pt", #
        "labels": ["Bacterialblight", "Brownspot", "Leafsmut"], #
        "num_classes": 3, #
        "fc_type": "custom_sequential_new_crop", # Specific custom sequential head for Rice
        "remedies": {
            "Bacterialblight": "Suggested remedies for Bacterial Blight in Rice: Remove infected leaves, apply copper fungicides. Ensure good air circulation.",
            "Brownspot": "Suggested remedies for Brown Spot in Rice: Use resistant varieties, apply appropriate fungicides, practice crop rotation.",
            "Leafsmut": "Suggested remedies for Leaf Smut in Rice: Use seed treatments, remove infected plant debris, consider systemic fungicides."
        }
    }
}

# Get model path, labels, num_classes, fc_type, and remedies for the selected crop
current_model_path = model_info[crop]["model_path"]
labels = model_info[crop]["labels"]
current_num_classes = model_info[crop]["num_classes"]
fc_type = model_info[crop]["fc_type"] # Get the FC type
static_remedies = model_info[crop]["remedies"]


@st.cache_resource
def load_and_configure_model(path, num_classes_for_model, fc_layer_type):
    model = resnet50(weights=None) # Start with a basic ResNet-50 without pretrained weights
    
    num_ftrs = model.fc.in_features # Get the input features for the FC layer

    if fc_layer_type == "single_linear":
        model.fc = nn.Linear(num_ftrs, num_classes_for_model)
    elif fc_layer_type == "custom_sequential_peanut":
        # Exact structure for Peanut's custom head (Linear 512, Dropout 0.3)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Specific dropout rate for Peanut
            nn.Linear(512, num_classes_for_model)
        )
    elif fc_layer_type == "custom_sequential_new_crop": # Now applies to "Rice"
        # Exact structure for Rice's custom head (Linear 256, Dropout 0.4)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes_for_model)
        )
    else:
        # This case should ideally not be reached if fc_type is correctly defined for all crops
        raise ValueError(f"Unknown FC layer type: {fc_layer_type} for crop {crop}. Please check model_info configuration.")

    # Load the state dictionary
    state_dict = torch.load(path, map_location=device)
    
    # Load the state dictionary into the model with the correctly configured FC layer
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

# Load the model for the selected crop, passing the fc_type
model = load_and_configure_model(current_model_path, current_num_classes, fc_type)

# Image transform - using the provided common transform (ImageNet mean/std)
# This transform should be consistent with how ALL your models were trained.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload an image of the leaf", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    uploaded_file = st.camera_input("Or take a picture")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, 1).item()
        prediction = labels[predicted_idx]

    st.success(f"🧠 Predicted Disease: **{prediction}**")

    # Display Static Remedies
    st.subheader("📋 Suggested Remedy (Static Advice):")
    # Check if the exact predicted class exists in static_remedies
    if prediction in static_remedies:
        # For Tomato, the remedies have 'description', 'mitigation', 'fertilizer' keys.
        # For Potato, they have 'Mitigation', 'Fertilizer', 'Fertilizer_Calendar'
        # For others and Rice, they are direct strings.
        remedy_info = static_remedies[prediction]
        if isinstance(remedy_info, dict): # Check if it's a dictionary (like Potato, Tomato)
            # Handle Tomato's specific keys
            if "description" in remedy_info:
                st.markdown(f"**📝 Description:** {remedy_info.get('description', 'N/A')}")
                st.markdown(f"**🛠️ Mitigation Tips:** {remedy_info.get('mitigation', 'N/A')}")
                st.markdown(f"**🌱 Fertilizer Recommendation:** {remedy_info.get('fertilizer', 'N/A')}")
            # Handle Potato's specific keys (Mitigation, Fertilizer, Fertilizer_Calendar)
            elif "Mitigation" in remedy_info: # This covers Potato
                st.markdown(remedy_info["Mitigation"])
                st.subheader("🌿 Fertilizer Recommendation")
                st.markdown(remedy_info["Fertilizer"])
                if 'Fertilizer_Calendar' in remedy_info:
                    st.subheader("📅 Fertilizer Calendar Plan")
                    st.markdown(remedy_info['Fertilizer_Calendar'])
        else: # It's a direct string (like Grapes, Peanut, and Rice)
            st.markdown(remedy_info)
    else:
        st.warning("No specific static remedy information available for this prediction.")


    # DeepSeek API (AI-generated remedy)
    st.subheader("🤖 AI-Powered Advice (DeepSeek)")
    # IMPORTANT: Store your API key securely, e.g., using Streamlit secrets
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

    if OPENROUTER_API_KEY and OPENROUTER_API_KEY.startswith("sk-"):
        if st.toggle("💬 Get AI-generated advice"):
            with st.spinner("Contacting AI expert..."):
                prompt = f"""
                My {crop} plant has {prediction}.
                Provide concise, practical advice for a farmer:
                - What is the main cause (1-2 sentences)?
                - What are quick fix or treatment steps?
                - Recommend specific fungicides/pesticides/fertilizers if applicable.
                - Provide 2-3 general tips for management or prevention.
                Keep the total response under 150 words.
                """
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek/deepseek-chat:free",
                    "messages": [
                        {"role": "system", "content": "You are a helpful agricultural expert providing practical advice."},
                        {"role": "user", "content": prompt}
                    ]
                }
                try:
                    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
                    if response.status_code == 200:
                        result = response.json()
                        reply = result['choices'][0]['message']['content']
                        st.info(reply)

                        if st.toggle("💬 Ask a follow-up question"):
                            user_follow_up_q = st.text_input("Ask your follow-up question here:")
                            if user_follow_up_q:
                                follow_up_payload = {
                                    "model": "deepseek/deepseek-chat:free",
                                    "messages": [
                                        {"role": "system", "content": "You are a helpful agricultural expert providing practical advice."},
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": reply},
                                        {"role": "user", "content": user_follow_up_q}
                                    ]
                                }
                                follow_up_resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(follow_up_payload))
                                if follow_up_resp.status_code == 200:
                                    st.info(follow_up_resp.json()['choices'][0]['message']['content'])
                                else:
                                    st.error(f"❌ Failed to get follow-up AI advice: {follow_up_resp.status_code} - {follow_up_resp.text}")
                    else:
                        st.error(f"❌ AI advice unavailable. Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ An error occurred while contacting the AI service: {e}")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")
    else:
        st.info("Please provide a valid OpenRouter API key to get AI-powered advice.")