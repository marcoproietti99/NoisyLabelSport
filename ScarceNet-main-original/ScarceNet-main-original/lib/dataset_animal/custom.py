from .ap10k import AnimalAP10KDataset

def custom(cfg, root, ann_file, is_train, transform):
    print(f"custom() called with ann_file: {ann_file}")
    """
    Dataset custom COCO-style compatibile con ScaRCeNet.
    
    :param cfg: Configurazione del training (cfg.DATASET.ROOT, ecc.)
    :param root: Cartella delle immagini
    :param ann_file: Path al file .json COCO delle annotazioni
    :param is_train: True se in training, False se in validazione/test
    :param transform: pipeline di trasformazioni (ToTensor, Normalize, ecc.)
    :return: Dataset pronto per il DataLoader
    """
    return AnimalAP10KDataset(cfg, root, image_set=ann_file, is_train=is_train, transform=transform)
