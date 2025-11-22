import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import timm
import os
from typing import Tuple

# --- Configuration et Constantes ---
MODEL_PATH = "best_model_finetuned.pth" 
IMAGE_SIZE = 224 # Taille d'entrée pour ResNet-50
NUM_CLASSES = 2
INPUT_FEATURES = 2048 # Nombre de features en sortie du ResNet-50 (avant la couche fc)

# Classes prédites par le modèle (doit correspondre à votre entraînement)
# Le notebook utilise la classe 'Real' en indice 0 et 'Fake' en indice 1.
CLASS_NAMES = ["Fake", "Real"] 

# --- Fonctions Utilitaires ---

# Reconstruction exacte de la tête de classification du notebook
def create_custom_classifier(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

@st.cache_resource
def load_model(model_path: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Charge le modèle ResNet-50 avec la tête personnalisée et ses poids entraînés.
    """
    if not os.path.exists(model_path):
        st.error(f"Erreur: Fichier modèle non trouvé à {model_path}. Veuillez vérifier le chemin.")
        # Retourner un modèle non fonctionnel
        model = timm.create_model('resnet50', pretrained=False)
        model.fc = nn.Identity() 
        return model

    try:
        # 1. Recréer l'architecture ResNet-50 de base SANS le classifieur final
        model = timm.create_model('resnet50', pretrained=False)
        
        # 2. Remplacer la couche 'fc' par la structure EXACTE utilisée à l'entraînement
        model.fc = create_custom_classifier(INPUT_FEATURES, num_classes)
        
        # 3. Charger les poids depuis le fichier .pth
        # map_location=torch.device('cpu') assure que le chargement fonctionne même sans GPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Le chargement doit être STRICT car la structure est maintenant correcte.
        model.load_state_dict(state_dict, strict=True) 
        
        model.eval()
        st.success("Modèle ResNet-50 Fine-tuned (Acc: 81.22%) chargé avec succès!")
        return model
    
    except Exception as e:
        # Afficher l'erreur pour le débogage (souvent un problème de chemin ou de clé)
        st.error(f"Erreur critique lors du chargement : La structure du state_dict n'a pas pu être restaurée.")
        st.exception(e)
        # Retourner un modèle non fonctionnel
        model = timm.create_model('resnet50', pretrained=False)
        model.fc = nn.Identity()
        return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Applique les transformations nécessaires à l'image avant l'inférence.
    (Redimensionnement, ToTensor, Normalisation)
    """
    # Utiliser les mêmes transformations de normalisation et de taille que le notebook
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(IMAGE_SIZE), # Le notebook utilise CenterCrop(224) pour le test
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Ajoute la dimension du batch (1)


def predict_image(model: nn.Module, image: Image.Image) -> Tuple[str, float, torch.Tensor]:
    """
    Effectue la prédiction sur l'image fournie.
    Retourne la classe prédite, le score de confiance, et le tenseur de probabilités complet.
    """
    # Préparation du tenseur d'entrée
    input_tensor = preprocess_image(image)
    
    # Inférence
    with torch.no_grad():
        outputs = model(input_tensor)
        
    # Obtention des probabilités (Softmax)
    probabilities = torch.softmax(outputs, dim=1)
    
    # Obtention de la prédiction la plus probable
    confidence, predicted_index = torch.max(probabilities, 1)
    
    # Extraction des résultats
    predicted_class = CLASS_NAMES[predicted_index.item()]
    confidence_score = confidence.item()
    
    # RETOURNE MAINTENANT LE TENSEUR DE PROBABILITÉS COMPLET
    return predicted_class, confidence_score, probabilities


# --- Chargement du Modèle ---
model = load_model(MODEL_PATH)

# --- Interface Utilisateur Streamlit ---
st.set_page_config(
    page_title="Détecteur de Deepfakes par Image",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("👁️ Détecteur de Deepfakes par Image")
st.markdown("---")

st.info(f"Cette application utilise votre modèle **ResNet-50 Fine-tuned** (Accuracy Finale : **81.22%**) pour classifier si un visage est réel ou généré (deepfake).")

# Widget de téléversement de fichier
uploaded_file = st.file_uploader(
    "Téléversez une image de visage (.jpg, .png)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Initialiser probabilities avant le bloc try/except pour garantir la portée
    probabilities = torch.tensor([[0.0, 0.0]])
    
    try:
        # Lecture de l'image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Image Téléversée', use_column_width=True)
            
        with col2:
            st.subheader("Analyse du Modèle")
            
            # Afficher un spinner pendant la prédiction
            with st.spinner('Analyse en cours...'):
                # CAPTURE probabilities ici
                predicted_class, confidence_score, probabilities = predict_image(model, image)
            
            # Affichage des résultats
            if predicted_class == "Fake":
                st.error(f"🔴 RÉSULTAT: L'image est classée comme **{predicted_class}**")
                st.metric(
                    label="Score de Confiance (Fake)",
                    value=f"{confidence_score:.2f}"
                )
            else:
                st.success(f"🟢 RÉSULTAT: L'image est classée comme **{predicted_class}**")
                st.metric(
                    label="Score de Confiance (Real)",
                    value=f"{confidence_score:.2f}"
                )

            # Ces lignes fonctionnent maintenant car probabilities est défini dans la portée locale
            st.write(f"Probabilité pour 'Fake': **{probabilities[0][1].item():.2f}**")
            st.write(f"Probabilité pour 'Real': **{probabilities[0][0].item():.2f}**")
            
        st.markdown("---")
        st.caption("Application développée dans le cadre du projet G2-Détection de vidéos falsifiées.")

    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement de l'image.")
        st.exception(e)

else:
    st.warning("Veuillez téléverser une image de visage pour commencer l'analyse.")