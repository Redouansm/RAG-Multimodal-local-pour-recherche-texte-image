# ============================================
# MULTIMODAL RAG SUR TON PC LOCAL
# ============================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
import open_clip
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIG LOCALE - A ADAPTER
# ============================================

# 1. Telecharge le dataset depuis Kaggle
# https://www.kaggle.com/datasets/nltkacademy/fashion-product-images-and-text-dataset
# Et place-le dans un dossier

# Exemple de structure:
# C:\Users\smahri\datasets\fashion_dataset\
#   ├── data.csv
#   └── data\
#       ├── image1.jpg
#       ├── image2.jpg
#       └── ...

# Change ces chemins selon ta structure
DATASET_DIR = r"C:\Users\smahri\datasets\fashion_dataset"  # A ADAPTER
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMAGES_DIR = os.path.join(DATASET_DIR, "data")

# GPU ou CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Device utilise: {DEVICE}")
print(f"[OK] GPU disponible: {torch.cuda.is_available()}\n")

# Verifie les chemins
print("[INFO] Verification des chemins:")
print(f"   CSV existe: {os.path.exists(CSV_PATH)}")
print(f"   Images dir existe: {os.path.exists(IMAGES_DIR)}\n")

if not os.path.exists(CSV_PATH):
    print(f"[ERREUR] CSV non trouve: {CSV_PATH}")
    print("   Telecharge le dataset depuis Kaggle et adapte le chemin\n")
    exit()

if not os.path.exists(IMAGES_DIR):
    print(f"[ERREUR] Dossier images non trouve: {IMAGES_DIR}")
    print("   Verifie la structure du dataset\n")
    exit()


# ============================================
# CLASSE MULTIMODAL RAG
# ============================================

class SimpleMultimodalRAG:
    def __init__(self, csv_path=CSV_PATH, images_dir=IMAGES_DIR, device=DEVICE):

        self.csv_path = csv_path
        self.images_dir = images_dir
        self.device = device
        
        print(f"[INFO] Chargement du CSV: {csv_path}")
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"   -> {len(self.df)} produits charges")
        print(f"   -> Colonnes: {list(self.df.columns)}\n")
        
        # Remplis NaN
        self.df = self.df.fillna("")
        
        # Detecte la colonne image
        image_col = None
        for col in self.df.columns:
            if 'image' in col.lower():
                image_col = col
                break
        
        if not image_col:
            print(f"[ERREUR] Aucune colonne 'image' trouvee!")
            print(f"   Colonnes disponibles: {list(self.df.columns)}")
            exit()
        
        print(f"[OK] Colonne image utilisee: '{image_col}'\n")
        
        # Cree les chemins d'images
        self.df["image_path"] = self.df[image_col].apply(
            lambda x: os.path.join(images_dir, str(x)) if pd.notna(x) else ""
        )
        
        # Filtre les images qui existent
        valid_images = self.df[self.df["image_path"].apply(lambda x: os.path.exists(x))]
        print(f"[OK] {len(valid_images)}/{len(self.df)} images valides\n")
        self.df = valid_images.reset_index(drop=True)

        # Load CLIP
        print("[INFO] Chargement du modele CLIP...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model.eval()
        print("   [OK] Modele charge\n")

        # Compute embeddings
        print("[INFO] Encodage des images...")
        self.image_embeddings = self._encode_all_images(batch_size=32)

        print("\n[INFO] Encodage des textes...")
        self.text_embeddings = self._encode_all_texts(batch_size=16)

        print("\n[OK] Multimodal RAG initialise!")
        print(f"   -> {len(self.df)} produits indexes\n")


    # ==============================
    # IMAGE ENCODING
    # ==============================
    def _encode_all_images(self, batch_size=32):
        paths = list(self.df["image_path"].values)
        all_embs = []
        
        print(f"[STAT] Total images: {len(paths)}")

        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch = []
                for p in paths[i:i + batch_size]:
                    try:
                        img = Image.open(p).convert("RGB")
                        img = self.clip_preprocess(img)
                    except Exception as e:
                        print(f"[WARN] Erreur avec {p}: {e}")
                        img = torch.zeros(3, 224, 224)
                    batch.append(img)

                batch = torch.stack(batch).to(self.device)
                emb = self.clip_model.encode_image(batch)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embs.append(emb.cpu().numpy())
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                if (i + batch_size) % (batch_size * 5) == 0:
                    print(f"   -> {min(i + batch_size, len(paths))}/{len(paths)} images traitees")

        return np.vstack(all_embs)


    def _encode_query_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.clip_model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        return emb.cpu().numpy()[0]


    # ==============================
    # TEXT ENCODING
    # ==============================
    def _encode_all_texts(self, batch_size=16):
        texts = []
        
        # Combine les colonnes texte disponibles
        text_cols = [col for col in self.df.columns if any(
            keyword in col.lower() for keyword in ['name', 'title', 'description', 'text', 'category']
        )]
        
        print(f"   Colonnes texte utilisees: {text_cols}")
        
        for _, row in self.df.iterrows():
            parts = []
            for col in text_cols:
                if pd.notna(row.get(col)):
                    parts.append(str(row[col]))
            txt = " ".join(parts) if parts else "fashion product"
            texts.append(txt)

        all_embs = []
        print(f"[STAT] Total textes: {len(texts)}")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                tokens = self.clip_tokenizer(batch_texts).to(self.device)
                emb = self.clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embs.append(emb.cpu().numpy())
                
                torch.cuda.empty_cache()
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    print(f"   -> {min(i + batch_size, len(texts))}/{len(texts)} textes traites")

        return np.vstack(all_embs)


    def _encode_query_text(self, query: str):
        with torch.no_grad():
            tokens = self.clip_tokenizer([query]).to(self.device)
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]


    # ==============================
    # MULTIMODAL RETRIEVAL
    # ==============================
    def retrieve_multimodal(self, query, is_image=False, top_k=5, w_img=0.5, w_txt=0.5):

        if is_image:
            q_img = self._encode_query_image(query).reshape(1, -1)
            query_text = "Fashion image query"
            q_txt = self._encode_query_text(query_text).reshape(1, -1)
        else:
            q_txt = self._encode_query_text(query).reshape(1, -1)
            q_img = q_txt

        # Compute similarities
        sim_img = cosine_similarity(q_img, self.image_embeddings)[0]
        sim_txt = cosine_similarity(q_txt, self.text_embeddings)[0]

        # Fusion multimodale
        sim_total = w_img * sim_img + w_txt * sim_txt

        top_idx = np.argsort(sim_total)[-top_k:][::-1]

        results = []
        for idx in top_idx:
            row = self.df.iloc[idx]
            result = {
                "similarity": float(sim_total[idx]),
                "image_path": row["image_path"]
            }
            # Ajoute les autres colonnes
            for col in self.df.columns:
                if col not in ["image_path"] and pd.notna(row.get(col)):
                    result[col] = row[col]
            
            results.append(result)

        return results


    def describe_image(self, image_path: str):
        print("\n[INFO] Recherche multimodale...")
        results = self.retrieve_multimodal(image_path, is_image=True, top_k=5)

        print("\n[RESULTAT] Produits les plus similaires:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. Similarite: {r['similarity']:.3f}")
            for col, val in r.items():
                if col not in ["similarity", "image_path"] and val:
                    print(f"   {col}: {str(val)[:80]}")
            print()

        return results


# ============================================
# MENU INTERACTIF
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("MULTIMODAL RAG - PC LOCAL")
    print("=" * 60 + "\n")
    
    # Initialisation
    rag = SimpleMultimodalRAG()
    
    while True:
        print("\n" + "=" * 60)
        print("MENU")
        print("=" * 60)
        print("1. Recherche par TEXTE")
        print("2. Recherche par IMAGE")
        print("3. Tester une image du dataset")
        print("4. Quitter\n")
        
        choix = input("Choix (1-4): ").strip()
        
        if choix == "1":
            query = input("\nEntrez votre requete: ").strip()
            if query:
                results = rag.retrieve_multimodal(query, is_image=False, top_k=5)
                print(f"\n[RESULTAT] Resultats pour: '{query}'\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. Similarite: {r['similarity']:.3f}")
                    for col, val in r.items():
                        if col not in ["similarity", "image_path"] and val:
                            print(f"   {col}: {str(val)[:80]}")
                    print()
        
        elif choix == "2":
            image_path = input("\nChemin de l'image: ").strip()
            if os.path.exists(image_path):
                rag.describe_image(image_path)
            else:
                print(f"[ERREUR] Image non trouvee: {image_path}")
        
        elif choix == "3":
            image_files = [f for f in os.listdir(rag.images_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if image_files:
                test_image = os.path.join(rag.images_dir, image_files[0])
                print(f"\n[INFO] Image test: {image_files[0]}\n")
                rag.describe_image(test_image)
            else:
                print("[ERREUR] Aucune image trouvee")
        
        elif choix == "4":
            print("\n[INFO] Au revoir!")
            break
        
        else:
            print("[ERREUR] Choix invalide")