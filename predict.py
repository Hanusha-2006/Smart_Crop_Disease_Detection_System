import tensorflow as tf
import numpy as np
from google.colab import drive, files
from PIL import Image
import matplotlib.pyplot as plt
import json
import io

drive_save_path = '/content/drive/MyDrive/Plant_Disease_Project_final/'

model = tf.keras.models.load_model(
    drive_save_path + 'plant_disease_hierarchical_final.keras'
)
print("✅ Model loaded!")

# ── Step 2: Load Class & Plant Names ─────────────────────────────────────────
with open(drive_save_path + 'class_names.json', 'r') as f:
    class_names = json.load(f)

with open(drive_save_path + 'plant_names.json', 'r') as f:
    plant_to_idx = json.load(f)

idx_to_plant = {v: k for k, v in plant_to_idx.items()}
print(f"✅ Classes loaded : {len(class_names)}")
print(f"✅ Plants loaded  : {len(idx_to_plant)}")

# ── Step 3: Build Plant → Disease Mapping ────────────────────────────────────
plant_disease_map = {}
for idx, name in class_names.items():
    plant = name.split('___')[0]
    if plant not in plant_disease_map:
        plant_disease_map[plant] = []
    plant_disease_map[plant].append(int(idx))
print("✅ Plant-Disease mapping ready!")

# ── Step 4: Predict Function ──────────────────────────────────────────────────
def predict(img_array):
    disease_pred, plant_pred = model.predict(img_array, verbose=0)

    # Plant result
    plant_idx  = np.argmax(plant_pred[0])
    plant_name = idx_to_plant[plant_idx]
    plant_conf = plant_pred[0][plant_idx] * 100

    # Filter disease to identified plant only
    valid_indices  = plant_disease_map.get(plant_name, [])
    filtered_probs = np.zeros_like(disease_pred[0])
    for idx in valid_indices:
        filtered_probs[idx] = disease_pred[0][idx]

    # Normalize
    total = np.sum(filtered_probs)
    if total > 0:
        filtered_probs = filtered_probs / total

    disease_idx  = np.argmax(filtered_probs)
    disease_name = class_names[str(disease_idx)]
    confidence   = filtered_probs[disease_idx] * 100
    disease_only = disease_name.split('___')[1] if '___' in disease_name else disease_name
    final_answer = f"{plant_name} — {disease_only}"

    # Top 3
    top3_indices = np.argsort(filtered_probs)[::-1][:3]
    top3         = [(class_names[str(i)].split('___')[1] if '___' in class_names[str(i)]
                     else class_names[str(i)], filtered_probs[i]*100)
                    for i in top3_indices if filtered_probs[i] > 0]

    return {
        'plant'       : plant_name,
        'plant_conf'  : plant_conf,
        'disease'     : disease_only,
        'confidence'  : confidence,
        'final_answer': final_answer,
        'status'      : "✅ Healthy"  if 'healthy' in disease_name.lower() else "⚠️ Diseased",
        'color'       : 'green'      if 'healthy' in disease_name.lower() else 'red',
        'top3'        : top3
    }

# ── Step 5: Preprocess ────────────────────────────────────────────────────────
def preprocess(image_data):
    img       = Image.open(io.BytesIO(image_data)).convert('RGB')
    img       = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# ── Step 6: Upload & Predict ──────────────────────────────────────────────────
print("\n📂 Upload your plant leaf images...")
uploaded = files.upload()
print(f"✅ {len(uploaded)} image(s) uploaded!")

all_results = []

for filename, image_data in uploaded.items():
    img, img_array = preprocess(image_data)
    result         = predict(img_array)
    result['filename'] = filename
    result['img']      = img
    all_results.append(result)

    # Display image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis('off')
    fig.text(
        0.5, 0.01,
        f"🎯 FINAL  : {result['final_answer']}\n"
        f"🌿 Plant  : {result['plant']} ({result['plant_conf']:.1f}%)\n"
        f"🦠 Disease: {result['disease']}\n"
        f"📊 Conf   : {result['confidence']:.2f}%\n"
        f"📌 Status : {result['status']}",
        ha='center', fontsize=10, fontweight='bold', color=result['color'],
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
    )
    plt.title(f"📁 {filename}", fontsize=10)
    plt.subplots_adjust(bottom=0.35)
    plt.show()

    # Print result
    print(f"\n{'='*55}")
    print(f"📁 Image         : {filename}")
    print(f"{'='*55}")
    print(f"🎯 FINAL ANSWER  : {result['final_answer']}")
    print(f"{'─'*55}")
    print(f"🌿 Plant         : {result['plant']} ({result['plant_conf']:.2f}%)")
    print(f"🦠 Disease       : {result['disease']}")
    print(f"📊 Confidence    : {result['confidence']:.2f}%")
    print(f"📌 Status        : {result['status']}")
    print(f"{'─'*55}")
    print(f"   Top 3 (filtered to {result['plant']} only):")
    for rank, (name, conf) in enumerate(result['top3'], 1):
        print(f"   {rank}. {name:<40} {conf:.2f}%")
    print(f"{'='*55}\n")

# ── Step 7: Summary ───────────────────────────────────────────────────────────
print("\n📊 FINAL SUMMARY")
print("=" * 70)
print(f"{'Image':<20} {'Final Answer':<35} {'Conf':>8}  {'Status'}")
print("=" * 70)
for r in all_results:
    print(f"{r['filename']:<20} {r['final_answer']:<35} {r['confidence']:>7.2f}%  {r['status']}")
print("=" * 70)
print(f"✅ Total Tested  : {len(all_results)}")
print(f"🌿 Healthy       : {sum(1 for r in all_results if 'Healthy' in r['status'])}")
print(f"⚠️  Diseased      : {sum(1 for r in all_results if 'Diseased' in r['status'])}")
