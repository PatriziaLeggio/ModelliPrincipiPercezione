import os
import datetime
import csv
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# --- PATCH DI SICUREZZA PER MAC (SSL FIX) ---
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ---------------------------------------------
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import joblib

# ==============================================================================
# 1. CONFIGURAZIONE
# ==============================================================================

PROJECT_ROOT = "/Users/salmaaseed/Desktop/PROGETTO_MdP_001"
DATASET_DIR = os.path.join(PROJECT_ROOT, "children_handwriting_dataset")
STUDENT_PHOTOS_DIR = os.path.join(PROJECT_ROOT, "student_photos")

EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "esperimenti")
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(EXPERIMENTS_ROOT, f"run_{TIMESTAMP}_TRANSFER")
LEADERBOARD_FILE = os.path.join(EXPERIMENTS_ROOT, "leaderboard_generale.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(OUTPUT_DIR, "Report_Scientifico_Transfer.pdf")
CNN_MODEL_FILE = os.path.join(OUTPUT_DIR, "resnet_model.pth")
SVM_MODEL_FILE = os.path.join(OUTPUT_DIR, "svm_model.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16        
EPOCHS = 30            
LEARNING_RATE = 0.0001 
KFOLDS = 5
IMG_SIZE = 128         

with open(os.path.join(OUTPUT_DIR, "config_used.txt"), "w") as f:
    f.write(f"Model: ResNet18 (Pretrained)\nImg: {IMG_SIZE}\nEpochs: {EPOCHS}\nLR: {LEARNING_RATE}\n")

# ==============================================================================
# 2. CLASSI E DATASET
# ==============================================================================

class HandwritingDataset(Dataset):
    def __init__(self, data_dir=None, image_paths=None, labels=None, transform=None):
        self.transform = transform
        if data_dir:
            self.image_paths = []
            self.labels = []
            for label_dir in ["normal", "disgrafia"]:
                class_dir = os.path.join(data_dir, label_dir)
                if not os.path.exists(class_dir): continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.labels.append(0 if label_dir == "normal" else 1)
        else:
            self.image_paths = image_paths
            self.labels = labels

    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None: return torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.tensor(0.0)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0
        img_tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0) 
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img_tensor, label

class TransferResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ORA IL DOWNLOAD FUNZIONERA' GRAZIE ALLA PATCH IN ALTO
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

def preprocess_photo(path, target_size=(IMG_SIZE, IMG_SIZE)):
    img = cv2.imread(path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        gray = gray[y:y+h, x:x+w]
        thresh = thresh[y:y+h, x:x+w]
        
    coords = np.column_stack(np.where(thresh > 0))
    if coords.any():
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    gray = cv2.resize(gray, target_size) / 255.0
    tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def extract_features_for_svm(image_paths):
    features = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        density = np.sum(thresh > 0) / (IMG_SIZE * IMG_SIZE)
        features.append(np.hstack([hist, density]))
    if len(features) == 0: return np.array([])
    return np.array(features, dtype=np.float32)

def classify_probabilities(probs, threshold_high=0.7, threshold_low=0.4):
    labels = []
    for p in probs:
        if p > threshold_high: labels.append("APPROFONDIMENTO")
        elif p > threshold_low: labels.append("MONITORAGGIO")
        else: labels.append("NELLA NORMA")
    return labels

def safe_text(text): return text.encode('latin-1', 'replace').decode('latin-1')

# ==============================================================================
# 3. GRAFICI E STORICO
# ==============================================================================

def plot_learning_curves(history, filename="learning_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(epochs, history['train_loss'], 'b-', label='Train'); plt.plot(epochs, history['val_loss'], 'r--', label='Val'); plt.title('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(epochs, history['train_acc'], 'b-', label='Train'); plt.plot(epochs, history['val_acc'], 'r--', label='Val'); plt.title('Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, filename)); plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.ylabel('Reale'); plt.xlabel('Predetto'); plt.title('Matrice Confusione'); plt.savefig(os.path.join(OUTPUT_DIR, filename)); plt.close()

def plot_boxplot(cnn_scores, svm_scores, filename="boxplot_accuracy.png"):
    plt.figure(figsize=(8, 6)); plt.boxplot([cnn_scores, svm_scores], tick_labels=["ResNet (Transfer)", "SVM"], showmeans=True); plt.ylabel("Accuracy"); plt.title("Stabilità Modelli"); plt.grid(axis="y", linestyle="--", alpha=0.6); plt.savefig(os.path.join(OUTPUT_DIR, filename)); plt.close()

def plot_history_trend(csv_file, output_filename="storico_esperimenti.png"):
    if not os.path.exists(csv_file): return False
    dates, acc_cnn, acc_svm = [], [], []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f); header = next(reader)
            idx_cnn, idx_svm, idx_date = header.index("Acc_CNN"), header.index("Acc_SVM"), header.index("Data")
            for row in reader:
                if not row: continue
                dates.append(row[idx_date].split('_')[1][:5])
                acc_cnn.append(float(row[idx_cnn])); acc_svm.append(float(row[idx_svm]))
        x_pos = range(len(dates))
        plt.figure(figsize=(10, 6))
        plt.plot(x_pos, acc_cnn, marker='o', label='Deep Learning', color='skyblue', linewidth=2)
        plt.plot(x_pos, acc_svm, marker='s', label='SVM', color='orange', linewidth=2, linestyle='--')
        plt.xticks(x_pos, dates, rotation=45); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, output_filename)); plt.close()
        return True
    except: return False

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================

if __name__ == "__main__":
    print(f"🚀 Avvio Pipeline TRANSFER LEARNING (ResNet18) in: {OUTPUT_DIR}")
    
    full_dataset = HandwritingDataset(DATASET_DIR)
    if len(full_dataset) == 0: print(f"❌ Nessun dato in {DATASET_DIR}"); exit()

    all_idx = np.arange(len(full_dataset))
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42, stratify=full_dataset.labels)
    train_dataset = Subset(full_dataset, train_idx); test_dataset = Subset(full_dataset, test_idx)
    print(f"📊 Dataset: {len(full_dataset)} img (Train: {len(train_idx)}, Test: {len(test_idx)})")

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.GaussianBlur(kernel_size=3),
    ])

    print("\n[1/6] Cross-Validation ResNet18...")
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    cnn_cv_scores = []
    X_tr_paths = [full_dataset.image_paths[i] for i in train_idx]; y_tr_labels = [full_dataset.labels[i] for i in train_idx]
    
    for t_idx, v_idx in kf.split(X_tr_paths):
        curr_tr = HandwritingDataset(image_paths=[X_tr_paths[i] for i in t_idx], labels=[y_tr_labels[i] for i in t_idx], transform=train_transform)
        curr_vl = HandwritingDataset(image_paths=[X_tr_paths[i] for i in v_idx], labels=[y_tr_labels[i] for i in v_idx], transform=None)
        
        tr_ldr = DataLoader(curr_tr, batch_size=BATCH_SIZE, shuffle=True); vl_ldr = DataLoader(curr_vl, batch_size=BATCH_SIZE, shuffle=False)
        model = TransferResNet().to(DEVICE); opt = optim.Adam(model.parameters(), lr=LEARNING_RATE); crit = nn.BCELoss()
        
        for _ in range(3):
            model.train()
            for X, y in tr_ldr:
                X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1); opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        
        model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for X, y in vl_ldr:
                X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1); preds = (model(X) > 0.5).float(); correct += (preds == y).sum().item(); total += y.size(0)
        cnn_cv_scores.append(correct/total)

    print("\n[2/6] Training Finale ResNet con Scheduler...")
    idx_tr, idx_vl = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
    ds_curve_tr = HandwritingDataset(image_paths=[full_dataset.image_paths[train_idx[i]] for i in idx_tr], labels=[full_dataset.labels[train_idx[i]] for i in idx_tr], transform=train_transform)
    ds_curve_vl = Subset(train_dataset, idx_vl)

    ldr_tr = DataLoader(ds_curve_tr, batch_size=BATCH_SIZE, shuffle=True); ldr_vl = DataLoader(ds_curve_vl, batch_size=BATCH_SIZE, shuffle=False)
    
    final_model = TransferResNet().to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        final_model.train(); r_loss=0; corr=0; tot=0
        for X, y in ldr_tr:
            X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1); optimizer.zero_grad(); out = final_model(X); loss = criterion(out, y); loss.backward(); optimizer.step()
            r_loss += loss.item()*X.size(0); corr += ((out>0.5)==y).sum().item(); tot += y.size(0)
        history['train_loss'].append(r_loss/tot); history['train_acc'].append(corr/tot)
        
        final_model.eval(); r_loss=0; corr=0; tot=0
        with torch.no_grad():
            for X, y in ldr_vl:
                X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1); out = final_model(X); loss = criterion(out, y)
                r_loss += loss.item()*X.size(0); corr += ((out>0.5)==y).sum().item(); tot += y.size(0)
        history['val_loss'].append(r_loss/tot); history['val_acc'].append(corr/tot)
        
        scheduler.step(history['val_loss'][-1])
        print(f"   Epoca {epoch+1} - Val Acc: {history['val_acc'][-1]:.3f} | Loss: {history['val_loss'][-1]:.4f}")

    plot_learning_curves(history); torch.save(final_model.state_dict(), CNN_MODEL_FILE)

    print("\n[3/6] Test Finale..."); ldr_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False); y_true, y_pred = [], []
    final_model.eval()
    with torch.no_grad():
        for X, y in ldr_test:
            X = X.to(DEVICE); preds = final_model(X).cpu().numpy().flatten(); y_true.extend(y.numpy()); y_pred.extend((preds>0.5).astype(int))
    plot_confusion_matrix_custom(y_true, y_pred)

    print("\n[4/6] SVM..."); X_svm = extract_features_for_svm([full_dataset.image_paths[i] for i in train_idx]); y_svm = [full_dataset.labels[i] for i in train_idx]
    svm_model = SVC(kernel='rbf', probability=True); svm_cv = cross_val_score(svm_model, X_svm, y_svm, cv=KFOLDS, scoring='accuracy'); svm_model.fit(X_svm, y_svm); joblib.dump(svm_model, SVM_MODEL_FILE); plot_boxplot(cnn_cv_scores, svm_cv)

    print("\n[5/6] Aggiornamento Storico...")
    row = [TIMESTAMP, EPOCHS, BATCH_SIZE, f"{np.mean(cnn_cv_scores):.4f}", f"{np.mean(svm_cv):.4f}", OUTPUT_DIR]
    exists = os.path.isfile(LEADERBOARD_FILE)
    with open(LEADERBOARD_FILE, 'a', newline='') as f:
        wr = csv.writer(f); 
        if not exists: wr.writerow(["Data", "Epochs", "Batch", "Acc_CNN", "Acc_SVM", "Path"])
        wr.writerow(row)
    has_history = plot_history_trend(LEADERBOARD_FILE)

    print("\n[6/6] Generazione PDF (Transfer Edition)...")
    st_imgs = []
    if os.path.exists(STUDENT_PHOTOS_DIR): st_imgs = [os.path.join(STUDENT_PHOTOS_DIR, f) for f in os.listdir(STUDENT_PHOTOS_DIR) if f.lower().endswith((".png", ".jpg"))]
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, safe_text(f"Report Disgrafia - {TIMESTAMP}"), 0, 1, "C"); pdf.set_font("Arial", "", 11); pdf.ln(5)
    if not st_imgs: pdf.cell(0, 10, "No photos.", 0, 1)
    else:
        for img_path in st_imgs:
            tensor = preprocess_photo(img_path).to(DEVICE)
            if tensor is None: continue
            cnn_prob = final_model(tensor).item()
            feat = extract_features_for_svm([img_path])
            svm_prob = svm_model.predict_proba(feat)[:, 1][0] if len(feat)>0 else 0
            c_lbl = classify_probabilities([cnn_prob])[0]; s_lbl = classify_probabilities([svm_prob])[0]
            line = f"File: {os.path.basename(img_path)}"; res_cnn = f"ResNet: {c_lbl} ({cnn_prob:.2f})"; res_svm = f"SVM: {s_lbl} ({svm_prob:.2f})"
            pdf.cell(0, 8, safe_text(line), 0, 1); pdf.set_font("Arial", "B", 11); pdf.cell(0, 8, safe_text(f"  {res_cnn}"), 0, 1); pdf.cell(0, 8, safe_text(f"  {res_svm}"), 0, 1); pdf.set_font("Arial", "", 11); pdf.ln(2)

    pdf.add_page(); pdf.set_font("Arial", "B", 14); pdf.cell(0, 10, safe_text("Appendice: Transfer Learning"), 0, 1, "C")
    grafici = ["learning_curves.png", "confusion_matrix.png", "boxplot_accuracy.png", "storico_esperimenti.png"]
    for g in grafici:
        path = os.path.join(OUTPUT_DIR, g)
        if os.path.exists(path): pdf.ln(5); pdf.image(path, w=170)
    try: pdf.output(REPORT_FILE); print(f"📄 PDF: {REPORT_FILE}")
    except Exception as e: print(f"❌ Errore PDF: {e}")
    print("✅ FINE.")