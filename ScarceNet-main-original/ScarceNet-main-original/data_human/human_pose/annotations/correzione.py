import json
import os

# === Percorsi ===input_json = "C://Users//marco//OneDrive//Desktop//Addestramento_scarcenet//ScarceNet-main-original//ScarceNet-main-original//data_human//human_pose//annotations//dataset_unificato.json"
output_json = "C://Users//marco//OneDrive//Desktop//Addestramento_scarcenet//ScarceNet-main-original//ScarceNet-main-original//data_human//human_pose//annotations//dataset_unificato_fixed.json"

new_prefix = "images_merged"  # <-- dove sono le immagini realmente

# === Carica il JSON COCO ===
with open(input_json, 'r') as f:
    coco = json.load(f)

# === Aggiorna tutti i file_name ===
for img in coco["images"]:
    filename = os.path.basename(img["file_name"])  # rimuove 'data/' o altro
    img["file_name"] = os.path.join(new_prefix, filename)

# === Salva il nuovo JSON ===
with open(output_json, 'w') as f:
    json.dump(coco, f)

print(f"✅ JSON aggiornato salvato in: {output_json}")
