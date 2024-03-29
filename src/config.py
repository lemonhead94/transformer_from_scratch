from pathlib import Path
from typing import Dict


def get_config():
    return {
        "batch_size": 8,
        "number_of_epochs": 20,
        "learning_rate": 10**-4,
        "sequence_length": 482,
        "embedding_dimension": 512,
        "source_language": "de",
        "target_language": "en",
        "model_folder": "weights",
        "preload": None,
        "tokenizer_file": "tokenizer_data/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "model_basename": "tmodel_",
    }


def get_weights_file_path(config: Dict[str, str], epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
