import json

# Percorso del file di annotazione
annotation_file = "dataset_unificato_aggiornato.json"

# Carica il file di annotazione
with open(annotation_file, "r") as f:
    data = json.load(f)

# Aggiorna i percorsi delle immagini
for image in data["images"]:
    image["file_name"] = f"data/{image['file_name']}"

# Salva il file aggiornato
with open("dataset_unificato_fixed.json", "w") as f:
    json.dump(data, f, indent=4)

print("File di annotazione aggiornato con percorsi corretti.")