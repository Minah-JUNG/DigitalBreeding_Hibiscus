## classification phenotypes using CNN models 
## version 7

import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

##################################################

# setting
data_path    = '/mnt/d/Documents/HibiscusPhenotype.xlsx'
image_folder = '/mnt/d/Documents/Results'
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 30

##################################################

# preprocessing
df = pd.read_excel(data_path)
df['image_path'] = df['Accession'].apply(lambda x: os.path.join(image_folder, f"{x}.png"))
imgs, types, colors = [], [], []
for _, r in df.iterrows():
    img = cv2.imread(r['image_path'])
    if img is None: continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    imgs.append(img)
    types.append(r['Type']); colors.append(r['Color'])
X = np.array(imgs, dtype='float32') / 255.0
le_type = LabelEncoder(); le_color = LabelEncoder()
Y_type  = to_categorical(le_type.fit_transform(types))
Y_color = to_categorical(le_color.fit_transform(colors))

# split traing sets
X_train, X_temp, yt_train, yt_temp, yc_train, yc_temp = train_test_split(
    X, Y_type, Y_color, test_size=0.3, random_state=42,
    stratify=le_type.transform(types)
)
X_val, X_test, yt_val, yt_test, yc_val, yc_test = train_test_split(
    X_temp, yt_temp, yc_temp, test_size=0.5, random_state=42,
    stratify=np.argmax(yt_temp, axis=1)
)

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9,1.1)
)

##################################################

##### models

# simple CNN
def build_baseline_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32,3,activation='relu',input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes,activation='softmax')
    ])
    return model

# ResNet50
def build_resnet(input_shape, num_classes):
    base = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=out)

# EfficientNetB0
def build_efficientnet(input_shape, num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=out)

# ViT
def build_vit(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    patches = layers.Conv2D(64, 16, strides=16)(inp)
    seq = layers.Reshape(((IMG_SIZE//16)**2, 64))(patches)
    x = layers.LayerNormalization()(seq)
    for _ in range(4):
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64//4)(x, x)
        x = layers.Add()([x, attn])
        y = layers.LayerNormalization()(x)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.Dense(64)(y)
        x = layers.Add()([x, y])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inp, outputs=out)

##################################################

# training
def train_and_evaluate(model_builder, optimizer_cls, lr, task_name, Y_train, Y_val, Y_test):
    print(f"\n>>> Training {model_builder.__name__} for {task_name}")
    model = model_builder((IMG_SIZE, IMG_SIZE, 3), Y_train.shape[1])
    # instantiate new optimizer per model
    optimizer = optimizer_cls(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)
    gen = datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
    steps = len(X_train) // BATCH_SIZE
    model.fit(
        gen,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        validation_data=(X_val, Y_val),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ],
        verbose=2
    )
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"{model_builder.__name__}-{task_name} Test Acc: {acc:.4f}")
    preds = np.argmax(model.predict(X_test), axis=1)
    trues = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=le_type.classes_ if task_name=='Type' else le_color.classes_,
        yticklabels=le_type.classes_ if task_name=='Type' else le_color.classes_,
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right')
    plt.title(f"{model_builder.__name__} {task_name} CM")
    plt.savefig(f"cm_{model_builder.__name__}_{task_name}.png", bbox_inches='tight')
    plt.close()
    return {'model': model_builder.__name__, 'task': task_name, 'accuracy': acc}


# comparison
model_configs = [
    (build_baseline_cnn, optimizers.Adam,       1e-3),
    (build_resnet,      optimizers.Nadam,       1e-4),
    (build_efficientnet,optimizers.Adam,       5e-5),
    (build_vit,         optimizers.Nadam,       2e-4)
]
results = []
for builder, opt_cls, lr in model_configs:
    results.append(train_and_evaluate(builder, opt_cls, lr, 'Type', yt_train, yt_val, yt_test))
    results.append(train_and_evaluate(builder, opt_cls, lr, 'Color', yc_train, yc_val, yc_test))

##################################################

# save 
res_df = pd.DataFrame(results)
print(res_df)
res_df.to_csv('comparison_results.csv', index=False)
print("===== Training complete. Comparison results saved. ===== ")

##################################################
