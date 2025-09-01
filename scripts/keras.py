import os, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

DATA_ROOT = os.environ.get("AMHCD_DIR", "/content/drive/MyDrive/Master/data/MNIST_CNN_TP") 
IMG_SIZE = (64,64); BATCH=32; NUM_CLASSES=34

os.makedirs("outputs", exist_ok=True)

train_ds = image_dataset_from_directory(
    DATA_ROOT, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)
val_ds = image_dataset_from_directory(
    DATA_ROOT, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

inputs = layers.Input(shape=(64,64,3))
x = layers.Conv2D(6,(5,5),activation='relu')(inputs)
x = layers.AveragePooling2D((2,2))(x)
x = layers.Conv2D(16,(5,5),activation='relu')(x)
x = layers.AveragePooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(120,activation='relu')(x)
x = layers.Dense(84,activation='relu')(x)
outputs = layers.Dense(NUM_CLASSES,activation='softmax')(x)
model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cbs=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True),
     tf.keras.callbacks.ModelCheckpoint("outputs/lenet5_amhcd_best.keras",
             monitor="val_accuracy",save_best_only=True)]

hist = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=cbs)

# Learning curves
plt.figure(); plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy'])
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(['train','val'])
plt.title('Accuracy vs Epoch'); plt.tight_layout()
plt.savefig('outputs/amhcd_fig_learning_curves.png', dpi=200)

# Confusion matrix on validation set
y_true, y_pred = [], []
for xb, yb in val_ds:
    p = model.predict(xb, verbose=0).argmax(1)
    y_pred.extend(p.tolist()); y_true.extend(yb.numpy().tolist())
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
plt.tight_layout(); plt.savefig('outputs/amhcd_fig_confusion_matrix.png', dpi=200)

# Per-class CSV
acc = (cm.diagonal() / np.maximum(cm.sum(1), 1))
class_names = train_ds.class_names
pd.DataFrame({'Character':class_names,'Accuracy':acc}).to_csv(
    'outputs/AMHCD_lenet5_per_class_accuracy.csv', index=False)

print(classification_report(y_true, y_pred, target_names=class_names))
print("Saved artifacts in outputs/")
