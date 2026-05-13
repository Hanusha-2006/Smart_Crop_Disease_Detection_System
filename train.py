import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
import os

IMG_SIZE        = 224
BATCH_SIZE      = 32
DATASET_PATH    = "/content/PlantVillage-Dataset-master/raw/color"
drive_save_path = '/content/drive/MyDrive/Plant_Disease_Project_final/'
os.makedirs(drive_save_path, exist_ok=True)

# ── Data Generators ──────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 25,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    brightness_range   = [0.7, 1.3],
    horizontal_flip    = True,
    zoom_range         = 0.2,
    validation_split   = 0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'training',
    shuffle     = True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'validation',
    shuffle     = False
)

NUM_CLASSES  = train_gen.num_classes
class_names  = list(train_gen.class_indices.keys())
plant_names  = sorted(list(set([c.split('___')[0] for c in class_names])))
plant_to_idx = {plant: idx for idx, plant in enumerate(plant_names)}
NUM_PLANTS   = len(plant_names)

print(f"✅ Training images   : {train_gen.samples}")
print(f"✅ Validation images : {val_gen.samples}")
print(f"✅ Total classes     : {NUM_CLASSES}")
print(f"✅ Total plants      : {NUM_PLANTS}")
print(f"✅ Plants            : {plant_names}")

# ── Save Class & Plant Names ──────────────────────────────────────────────────
with open(os.path.join(drive_save_path, 'class_names.json'), 'w') as f:
    json.dump({v: k for k, v in train_gen.class_indices.items()}, f)
with open(os.path.join(drive_save_path, 'plant_names.json'), 'w') as f:
    json.dump(plant_to_idx, f)
print("✅ Class & plant names saved!")

# ── Custom Generator ──────────────────────────────────────────────────────────
class PlantDiseaseGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, class_names, plant_to_idx):
        self.generator    = generator
        self.class_names  = class_names
        self.plant_to_idx = plant_to_idx
        self.num_plants   = len(plant_to_idx)

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        images, disease_labels = self.generator[idx]
        plant_labels = np.zeros(
            (len(disease_labels), self.num_plants),
            dtype=np.float32
        )
        for i, disease_one_hot in enumerate(disease_labels):
            disease_idx  = np.argmax(disease_one_hot)
            disease_name = self.class_names[disease_idx]
            plant_name   = disease_name.split('___')[0]
            plant_idx    = self.plant_to_idx.get(plant_name, 0)
            plant_labels[i, plant_idx] = 1.0

        return images, {
            'disease_output': disease_labels,
            'plant_output'  : plant_labels
        }

train_wrapped = PlantDiseaseGenerator(train_gen, class_names, plant_to_idx)
val_wrapped   = PlantDiseaseGenerator(val_gen,   class_names, plant_to_idx)

# ── Build Hierarchical Model ──────────────────────────────────────────────────
#
#   Image Input
#       │
#   MobileNetV2 (Frozen)
#       │
#   GlobalAveragePooling2D
#       │
#   ┌───┴──────────────────┐
#   │                      │
# Plant Branch         Image Features
# Dense(128)               │
# BN → Dropout             │
# Dense(64)                │
#   │                      │
# Plant Output     ┌───────┘
# (14 plants)      │
#   │              │
#   └──────┬───────┘
#          │
#     Concatenate
#          │
#     Dense(512)
#     BN → Dropout
#     Dense(256)
#     BN
#          │
#     Disease Output
#     (38 diseases)
#

def build_hierarchical_model():
    image_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')

    # ── Frozen Base ───────────────────────────────────────────────────────────
    base_model = tf.keras.applications.MobileNetV2(
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
        include_top = False,
        weights     = 'imagenet'
    )
    base_model.trainable = False

    base_output = base_model(image_input, training=False)
    pooled      = layers.GlobalAveragePooling2D()(base_output)

    # ── Plant Branch — Identifies Plant First ─────────────────────────────────
    plant_x = layers.Dense(128, activation='relu',
                           kernel_initializer='he_normal',
                           name='plant_dense1')(pooled)
    plant_x = layers.BatchNormalization(name='plant_bn1')(plant_x)
    plant_x = layers.Dropout(0.3, name='plant_dropout')(plant_x)
    plant_x = layers.Dense(64, activation='relu',
                           kernel_initializer='he_normal',
                           name='plant_dense2')(plant_x)
    plant_x = layers.BatchNormalization(name='plant_bn2')(plant_x)

    # Plant output — what plant is this?
    plant_output = layers.Dense(
        NUM_PLANTS,
        activation = 'softmax',
        name       = 'plant_output'
    )(plant_x)

    # ── ✅ KEY FIX: Feed Plant Info Into Disease Branch ───────────────────────
    # Concatenate image features + plant branch features
    # Disease branch now KNOWS which plant it is looking at
    disease_input = layers.Concatenate(name='disease_input_combine')(
        [pooled, plant_x]                # ✅ pooled features + plant features
    )

    # ── Disease Branch — Identifies Disease Based on Plant ────────────────────
    disease_x = layers.Dense(512, activation='relu',
                             kernel_initializer='he_normal',
                             name='disease_dense1')(disease_input)
    disease_x = layers.BatchNormalization(name='disease_bn1')(disease_x)
    disease_x = layers.Dropout(0.3, name='disease_dropout')(disease_x)
    disease_x = layers.Dense(256, activation='relu',
                             kernel_initializer='he_normal',
                             name='disease_dense2')(disease_x)
    disease_x = layers.BatchNormalization(name='disease_bn2')(disease_x)
    disease_x = layers.Dropout(0.2, name='disease_dropout2')(disease_x)
    disease_x = layers.Dense(128, activation='relu',
                             kernel_initializer='he_normal',
                             name='disease_dense3')(disease_x)
    disease_x = layers.BatchNormalization(name='disease_bn3')(disease_x)

    # Disease output — what disease does this plant have?
    disease_output = layers.Dense(
        NUM_CLASSES,
        activation = 'softmax',
        name       = 'disease_output'
    )(disease_x)

    model = tf.keras.Model(
        inputs  = image_input,
        outputs = [disease_output, plant_output],
        name    = 'PlantDisease_Hierarchical'
    )
    return model

model = build_hierarchical_model()
model.summary()

# ── Verify Initial Weights ────────────────────────────────────────────────────
print("\n🔍 Initial weight std BEFORE training:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        std = np.std(weights[0])
        print(f"   {layer.name:<35} std: {std:.4f}")

# ── Compile ───────────────────────────────────────────────────────────────────
model.compile(
    optimizer    = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss         = {
        'disease_output': 'categorical_crossentropy',
        'plant_output'  : 'categorical_crossentropy'
    },
    loss_weights = {
        'disease_output': 1.0,   # Disease is main goal
        'plant_output'  : 0.5    # ✅ Higher weight — plant must be correct first
    },
    metrics      = {
        'disease_output': 'accuracy',
        'plant_output'  : 'accuracy'
    }
)
# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    # ✅ Saves best model in .keras format
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = os.path.join(drive_save_path, 'best_hierarchical.keras'),
        monitor        = 'val_disease_output_accuracy',
        save_best_only = True,
        mode           = 'max',
        verbose        = 1
    ),
    # ✅ Saves best model in .h5 format
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = os.path.join(drive_save_path, 'best_hierarchical.h5'),
        monitor        = 'val_disease_output_accuracy',
        save_best_only = True,
        mode           = 'max',
        verbose        = 1
    ),
    # ✅ Stops when model stops learning
    tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_disease_output_accuracy',
        patience             = 3,
        restore_best_weights = True,
        mode                 = 'max',
        verbose              = 1
    ),
    # ✅ Reduces LR when stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.2,
        patience = 2,
        min_lr   = 1e-6,
        verbose  = 1
    )
]


# ── Train ─────────────────────────────────────────────────────────────────────
print("\n🚀 Training Hierarchical Model...")
print("   Plant branch trains first → Disease branch learns from plant")
history = model.fit(
    train_wrapped,
    validation_data = val_wrapped,
    epochs          = 7,
    callbacks       = callbacks
)
# ── Verify Weights AFTER Training ────────────────────────────────────────────
print("\n🔍 Weight std AFTER training:")
print("=" * 55)
all_ok = True
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        std    = np.std(weights[0])
        status = "✅ Trained" if std > 0.05 else "❌ NOT Trained"
        if std < 0.05:
            all_ok = False
        print(f"   {layer.name:<35} std: {std:.4f}  {status}")
print("=" * 55)
if all_ok:
    print("✅ All layers trained correctly!")
else:
    print("❌ Some layers not trained!")

# ── Plot Results ──────────────────────────────────────────────────────────────
acc      = history.history['disease_output_accuracy']
val_acc  = history.history['val_disease_output_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc,     label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Disease Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss,     label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(drive_save_path, 'training_plot.png'))
plt.show()

print(f"\n✅ Final Train Accuracy : {acc[-1]*100:.2f}%")
print(f"✅ Final Val Accuracy   : {val_acc[-1]*100:.2f}%")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
Y_pred = model.predict(val_wrapped)
y_pred = np.argmax(Y_pred[0], axis=1)
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Greens')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(
    y_true, y_pred,
    target_names=val_gen.class_indices.keys()
))

# ── Save Both Formats — Best Epoch ───────────────────────────────────────────

# ✅ Load best model saved by ModelCheckpoint
best_model = tf.keras.models.load_model(
    os.path.join(drive_save_path, 'best_hierarchical.keras')
)
print("✅ Best model loaded from callback!")

# ✅ Save best model in both formats
keras_path = os.path.join(drive_save_path, 'plant_disease_hierarchical_final.keras')
h5_path    = os.path.join(drive_save_path, 'plant_disease_hierarchical_final.h5')

best_model.save(keras_path)
print(f"✅ Saved .keras : {keras_path}")

best_model.save(h5_path)
print(f"✅ Saved .h5    : {h5_path}")

# ── Verify ────────────────────────────────────────────────────────────────────
print("\n📋 Verification:")
for path in [keras_path, h5_path]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ {os.path.basename(path):<50} {size_mb:.2f} MB — SAVED IN DRIVE")
    else:
        print(f"   ❌ {os.path.basename(path)} — NOT SAVED!")
