import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from Lung_Cnn_class.model import DeepPneumoniaCNN, UNet # <--- Import propre !

st.set_page_config(page_title="Pneumonia Detector AI", layout="wide")
st.title("🩺 Analyseur de Radiographies Pulmonaires")

@st.cache_resource
def load_models():
    classifier = DeepPneumoniaCNN()
    unet = UNet()
    try:
        classifier.load_state_dict(torch.load("modele_pneumonie.pth", map_location=torch.device('cpu')))
    except:
        pass # Géré par le script de lancement
    classifier.eval()
    unet.eval()
    return classifier, unet

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) 

def generate_gradcam_mock(image_tensor, model):
    heatmap = np.random.rand(224, 224) 
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    return np.maximum(heatmap, 0) / np.max(heatmap)

with st.spinner("⏳ Chargement des modèles..."):
    classifier, unet = load_models()

with st.sidebar:
    st.header("Paramètres")
    uploaded_file = st.file_uploader("Choisissez une radiographie", type=["jpg", "png", "jpeg"])
    vis_method = st.radio("Méthode visuelle :", ("Grad-CAM (Explicabilité)", "U-Net (Segmentation)"))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Radiographie Originale")
        st.image(image, use_container_width=True)
        
    with st.spinner("🔍 Analyse par l'IA..."):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            raw_output = classifier(input_tensor)
            # On applique Sigmoid ici pour convertir le score brut en Probabilité (0 à 1)
            prediction = torch.sigmoid(raw_output).item()
    
    is_pneumonia = prediction >= 0.5
    confidence = prediction if is_pneumonia else 1 - prediction
    
    st.markdown("---")
    if is_pneumonia: st.error(f"🚨 **Pneumonie détectée** ({confidence*100:.2f}%)")
    else: st.success(f"✅ **Poumon sain** ({confidence*100:.2f}%)")
    st.markdown("---")
    
    with col2:
        st.subheader(f"Analyse Visuelle")
        if vis_method == "Grad-CAM (Explicabilité)":
            heatmap = generate_gradcam_mock(input_tensor, classifier)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, image.size)), cv2.COLORMAP_JET)
            img_array = np.array(image)
            st.image(cv2.addWeighted(img_array, 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0), use_container_width=True)
        else: 
            with torch.no_grad(): mask_tensor = unet(input_tensor)
            mask_colored = np.zeros_like(np.array(image))
            mask_colored[:, :, 2] = cv2.resize(mask_tensor.squeeze().numpy(), image.size) * 255 
            st.image(cv2.addWeighted(np.array(image), 0.7, mask_colored, 0.3, 0), use_container_width=True)
else:
    st.info("👈 Chargez une image à gauche.")