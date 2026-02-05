import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image

CAMINHO_DATASET = 'dataset/Images' 
CAMINHO_MODELO = 'modelo_cachorros.keras' 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def criar_e_treinar_modelo():

    treino = image_dataset_from_directory(
        CAMINHO_DATASET, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    validacao = image_dataset_from_directory(
        CAMINHO_DATASET, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    racas = treino.class_names
    print(f"Raças encontradas: {len(racas)}")

    modelo_base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    modelo_base.trainable = False

    modelo = models.Sequential([
        layers.Rescaling(1./127.5, offset=-1, input_shape=(224, 224, 3)),
        modelo_base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(racas), activation='softmax')
    ])

    modelo.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    parada_rapida = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    modelo.fit(treino, validation_data=validacao, epochs=10, callbacks=[parada_rapida])
    
    modelo.save(CAMINHO_MODELO)
    print(f"Modelo salvo em: {CAMINHO_MODELO}")
    
    return modelo, racas

def carregar_classes():
   
    if os.path.exists(CAMINHO_DATASET):
        return sorted([d for d in os.listdir(CAMINHO_DATASET) if os.path.isdir(os.path.join(CAMINHO_DATASET, d))])
    return []
def predizer_raca(model, racas, img_path):
    if not os.path.exists(img_path):
        print("imagem nao encontrada")
        return
    
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    raca = racas[np.argmax(score)]
    confianca = 100 * np.max(score)

    print(f"--- RESULTADO ---")
    print(f"Raça: {raca}")
    print(f"Confiança: {confianca:.2f}%")
    
    plt.imshow(img)
    plt.title(f"{raca} ({confianca:.2f}%)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    
    if os.path.exists(CAMINHO_MODELO):
        print("Modelo encontrado! Carregando...")
        model = tf.keras.models.load_model(CAMINHO_MODELO)
        racas = carregar_classes()
    else:
        print("Modelo não encontrado. Vamos treinar do zero.")
        model, racas = criar_e_treinar_modelo()

    if model:
        while True:
            caminho_img = input("\nDigite o caminho da imagem (ou 'sair' para encerrar): ")
            if caminho_img.lower() == 'sair':
                break
            # Remove aspas caso o usuário copie o caminho como "C:\..."
            caminho_img = caminho_img.replace('"', '') 
            predizer_raca(model, racas, caminho_img)