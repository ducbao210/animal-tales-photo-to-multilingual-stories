import os
from pathlib import Path

from speciesnet.classifier import SpeciesNetClassifier

from src.animal_scan.config import load_config

config = load_config(Path(__file__).resolve().parent.parent / "configs.yaml")
model_name = config.values["speciesnet"]["model_id"]

print("Loading SpeciesNet Model...")

classifier = SpeciesNetClassifier(model_name=model_name, device="cpu")

labels_file_path = classifier.model_info.classifier_labels
print(f"Found label path: {labels_file_path}")

output_txt = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "speciesnet_labels_extracted.txt"
)

with open(labels_file_path, "r", encoding="utf-8") as fp:
    labels = [line.strip() for line in fp if line.strip()]

with open(output_txt, "w", encoding="utf-8") as fp:
    for idx, label in enumerate(labels):
        fp.write(f"{idx}: {label}\n")

print(f"Extracted {len(labels)} labels.")
print(f"Result path: {output_txt}")
