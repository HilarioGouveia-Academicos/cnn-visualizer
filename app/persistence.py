"""Find and load dataset checkpoints without downloading training data."""
import json
from datetime import datetime, timezone
from uuid import uuid4
from copy import deepcopy
from pathlib import Path

import tensorflow as tf

from app.datasets import DATASETS
from app.models import validate_architecture
from app.state import architecture_config, configuration_id

SAVED_MODELS = Path(__file__).resolve().parents[1] / "saved_models"


def checkpoint_path(config):
    return SAVED_MODELS / (configuration_id(config) + ".keras")


def load_training(config):
    path = checkpoint_path(config)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["config"] != architecture_config(config):
        raise ValueError("O modelo salvo pertence a outra configuração.")
    model = tf.keras.models.load_model(path, compile=False)
    model.get_layer("classification_logits")
    return model, metadata["history"]


def save_training(config, model, history):
    path = checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    path.with_suffix(".json").write_text(json.dumps({
        "config": architecture_config(config), "history": history,
    }), encoding="utf-8")


def execution_record(config, history, duration_seconds, total_images):
    """Snapshot actual run settings, independent of later widget edits."""
    training_count = int(total_images * 0.85)
    return {
        "schema_version": 1, "run_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": architecture_config(config), "history": deepcopy(history),
        "training": {"epochs_requested": config["epochs"],
                     "epochs_completed": len(history["loss"]),
                     "batch_size": config["batch_size"], "seed": config.get("seed", 42),
                     "total_images": total_images, "training_images": training_count,
                     "validation_images": total_images - training_count,
                     "validation_split": 0.15, "split_method": "keras_tail_before_shuffle",
                     "optimizer": "adam", "learning_rate": 0.001,
                     "loss": "sparse_categorical_crossentropy"},
        "duration_seconds": duration_seconds,
        "metrics": {"final_val_accuracy": history["val_accuracy"][-1],
                    "best_val_accuracy": max(history["val_accuracy"]),
                    "final_val_loss": history["val_loss"][-1]},
        "weights": "final_epoch",
    }


def save_execution(record, model, name):
    """Publish metadata only after the model is saved; never overwrite a run."""
    name = name.strip()
    if not name:
        raise ValueError("Informe um nome para a execução.")
    run_id = record["run_id"]
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Identificador de execução inválido.")
    directory = SAVED_MODELS / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        raise ValueError("Esta execução já foi salva.")
    metadata = deepcopy(record)
    metadata["name"] = name
    model.save(directory / "model.keras")
    temporary = directory / "metadata.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata


def saved_executions(dataset):
    records = []
    for path in SAVED_MODELS.glob("runs/*/metadata.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            datetime.fromisoformat(record["created_at"])
            if record["run_id"] != path.parent.name or not record["name"].strip():
                continue
            if record["config"]["dataset"] == dataset and (path.parent / "model.keras").exists():
                records.append((path.parent / "model.keras", record))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(records, key=lambda item: item[1]["created_at"], reverse=True)


def load_execution(path):
    path = Path(path)
    record = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(path, compile=False)
    model.get_layer("classification_logits")
    return model, record


def latest_training(dataset):
    """Return the newest valid training for this dataset and any load warnings."""
    errors = []
    candidates = [(path, None, path.stat().st_mtime) for path in SAVED_MODELS.glob("*.keras")]
    candidates += [(path, record, datetime.fromisoformat(record["created_at"]).timestamp())
                   for path, record in saved_executions(dataset)]
    candidates.sort(key=lambda item: item[2], reverse=True)
    for path, record, _ in candidates:
        try:
            metadata = record or json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            config = metadata["config"]
            if config["dataset"] != dataset:
                continue
            config = architecture_config(config)
            if record is None and configuration_id(config) != path.stem:
                raise ValueError("Identificador incompatível com a configuração salva.")
            validate_architecture(config["num_conv"], (config["kernel"],) * 2,
                                  DATASETS[dataset]["shape"])
            history = metadata["history"]
            lengths = [len(history[key]) for key in
                       ("loss", "val_loss", "accuracy", "val_accuracy")]
            if not lengths[0] or len(set(lengths)) != 1:
                raise ValueError("Histórico de treinamento inválido.")
            model = tf.keras.models.load_model(path, compile=False)
            if tuple(model.input_shape[1:]) != DATASETS[dataset]["shape"]:
                raise ValueError("Formato de entrada incompatível com o dataset.")
            model.get_layer("classification_logits")
            for index in range(config["num_conv"]):
                model.get_layer(f"conv_{index + 1}")
            return (model, config, history), errors
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Não foi possível recuperar {path.name}: {exc}")
    return None, errors
