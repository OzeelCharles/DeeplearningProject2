import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kagglehub
from tqdm import tqdm
from Lung_Cnn_class.model import DeepPneumoniaCNN # Import de ton architecture

MODEL_PATH = "modele_pneumonie.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_save():
    print(f" GPU détecté : {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "⚠️️ CPU utilisé")
    path = os.path.join(os.getcwd(), '..')
    lungs_folder_train = os.path.join(path, "chest_xray", "train")
    lungs_folder_test = os.path.join(path, "chest_xray", "test")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(datasets.ImageFolder(lungs_folder_train, transform=transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(datasets.ImageFolder(lungs_folder_test, transform=transform), batch_size=32, shuffle=False)

    model = DeepPneumoniaCNN().to(device)
    
    # PARAMÈTRES CORRIGÉS : 
    # 1. BCEWithLogitsLoss gère mathématiquement le Sigmoid pour éviter les blocages à 0.5
    criterion = nn.BCEWithLogitsLoss() 
    # 2. Learning rate plus bas (1e-4) avec AdamW pour une meilleure régularisation
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", colour='green')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # Avec BCEWithLogitsLoss, le seuil de décision mathématique est 0.0 (et non 0.5)
            predicted = (outputs >= 0.0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix(loss=f"{train_loss/total:.4f}", acc=f"{correct/total:.4f}")

    print("\n✅ Entraînement terminé ! Sauvegarde des poids...")
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    # La logique "Auto-Launch"
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Modèle introuvable sur le disque.")
        print("⚙️ Lancement de la phase d'entraînement (Ceci ne se produira qu'une seule fois)...")
        train_and_save()
    else:
        print("✅ Modèle trouvé en mémoire ! Phase d'entraînement ignorée.")
        
    print("🌐 Démarrage du serveur Streamlit...")
    
    # Lance la commande console automatiquement
    os.system("streamlit run app.py")