import json
import os

input_json = "C://Users//marco//OneDrive//Desktop//Addestramento_scarcenet//ScarceNet-main-original//ScarceNet-main-original//data_human//human_pose//annotations//dataset_unificato_fixed2.json"
output_json = "C://Users//marco//OneDrive//Desktop//Addestramento_scarcenet//ScarceNet-main-original//ScarceNet-main-original//data_human//human_pose//annotations//dataset_unificato_fixed3.json"

with open(input_json, "r") as f:
    data = json.load(f)

for image in data["images"]:
    filename = os.path.basename(image["file_name"])
    image["file_name"] = os.path.join("images_merged", filename).replace("\\", "/")

with open(output_json, "w") as f:
    json.dump(data, f)

print(f"✅ File corretto salvato in: {output_json}")