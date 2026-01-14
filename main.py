import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split    
from sklearn.svm import SVC


DATASET_PATH = "dataset"
CATEGORIAS = ["gatos", "cachorros"]
IMG_SIZE = (64,64)

dados = []
labels = []

for categoria_idx, categoria in enumerate(CATEGORIAS):
    path = os.path.join(DATASET_PATH, categoria)
    if not os.path.exists(path):
        print(f"erro a pasta {path} nao existe")
        continue

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = imread(img_path)
            img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
            dados.append(img_resized.flatten())
            labels.append(categoria_idx)
        except Exception as e:
            print(f"erro ao processar a imagem {img_path}: {e}")

dados = np.array(dados)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(dados, labels, test_size=0.2, random_state=42)
model = SVC(kernel='poly',degree=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print(f"\nExemplo de teste:")
image_path = "images (1).jfif"

try:
    img = imread(image_path)
    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
    img_flattened = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_flattened)
    categoria_predita = CATEGORIAS[prediction[0]]
    print(f"A imagem foi classificada como: {categoria_predita}")
except Exception as e:
    print(f"erro ao processar a imagem de teste {image_path}: {e}")

 


