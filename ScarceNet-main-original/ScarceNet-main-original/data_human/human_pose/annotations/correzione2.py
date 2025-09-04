import json
import os

# === Config ===
input_json = "C://Users//marco//OneDrive//Desktop//Addestramento_scarcenet//ScarceNet-main-original//ScarceNet-main-original//data_human//human_pose//annotations//dataset_unificato_fixed.json"
output_json = "C://Users//marco//OneDrive//Desktop//Addestramento_scarcenet//ScarceNet-main-original//ScarceNet-main-original//data_human//human_pose//annotations//dataset_unificato_fixed2.json"

with open(input_json, "r") as f:
    coco = json.load(f)

for img in coco["images"]:
    filename = img["file_name"]

    # Rimuove prefissi strani come "data/", "annotations/data/", ecc.
    basename = os.path.basename(filename)

    # Usa solo la cartella corretta dove sono le immagini
    img["file_name"] = os.path.join("images_merged", basename).replace("\\", "/")

with open(output_json, "w") as f:
    json.dump(coco, f)

print(f"✅ File JSON pulito salvato in: {output_json}")
