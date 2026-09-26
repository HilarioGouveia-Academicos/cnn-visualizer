"""Settings kept separately from temporary Streamlit widgets."""
import hashlib
import json

DEFAULTS = {
    "dataset": "MNIST", "num_conv": 2, "filters": 32, "kernel": 3,
    "activation": "relu", "pooling": "MaxPooling2D", "dense_units": 64,
    "epochs": 3, "batch_size": 64, "seed": 42, "image_index": 0, "layer_index": 0,
    "filter_page": 1, "opacity": 0.45, "view": "Camadas", "sizing": "Balanced",
    "reversed_view": False, "flat_view": False, "image_source": "Dataset",
}
ARCHITECTURE_KEYS = ("dataset", "num_conv", "filters", "kernel", "activation",
                     "pooling", "dense_units")


def architecture_config(settings):
    return {key: settings[key] for key in ARCHITECTURE_KEYS}


def configuration_id(settings):
    payload = json.dumps(architecture_config(settings), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def initialize(ss):
    ss.setdefault("settings", DEFAULTS.copy())
    for setting, default in DEFAULTS.items():
        ss.settings.setdefault(setting, default)
    ss.setdefault("stage", "Construir")
    ss.setdefault("trained", False)
    ss.setdefault("ever_trained", False)
    ss.setdefault("history", {})
    ss.setdefault("upload_bytes", None)
    ss.setdefault("upload_name", "")
    ss.setdefault("session_runs", {})
    return ss.settings


def restore_dataset(ss):
    from app.persistence import latest_training
    cfg = ss.settings
    if ss.pop("restore_dataset", False) or "model_signature" not in ss:
        restored, errors = latest_training(cfg["dataset"])
        ss.restore_errors = errors
        ss.pop("restored_dataset", None)
        if restored is not None:
            restored_model, restored_config, restored_history = restored
            cfg.update(restored_config)
            ss.model = restored_model
            ss.model_signature = configuration_id(cfg)
            ss.history = restored_history
            ss.trained = ss.ever_trained = True
            ss.restored_dataset = cfg["dataset"]
            ss.pop("cam_result", None)



def synchronize_model(ss):
    from app.models import validate_architecture, model_from_config
    from app.datasets import DATASETS
    cfg = ss.settings
    signature = configuration_id(cfg)
    validate_architecture(cfg["num_conv"], (cfg["kernel"],) * 2, DATASETS[cfg["dataset"]]["shape"])
    if ss.get("model_signature") != signature:
        ss.model = model_from_config(cfg)
        ss.model_signature = signature
        ss.trained = False
        ss.history = {}
        ss.pop("restored_dataset", None)
        ss.pop("cam_result", None)


def accept_training(ss, model, history):
    ss.model = model
    ss.model_signature = configuration_id(ss.settings)
    ss.history = history
    ss.trained = ss.ever_trained = True
    ss.pop("restored_dataset", None)
    ss.pop("cam_result", None)


def register_execution(ss, model, record):
    ss.session_runs[record["run_id"]] = {"model": model, "record": record, "saved": False}


def navigate(ss, stage):
    ss.stage = stage


def remember(ss, name):
    ss.settings[name] = ss["control_" + name]
    if name == "dataset":
        ss.restore_dataset = True
        ss.settings["image_index"] = 0
        ss.settings["layer_index"] = 0
        ss.pop("control_image_index", None)
        ss.pop("control_layer_index", None)
    if name != "opacity":
        ss.pop("cam_result", None)


def keep_upload(ss):
    upload = ss.get("upload_widget")
    ss.upload_bytes = upload.getvalue() if upload is not None else None
    ss.upload_name = upload.name if upload is not None else ""
    ss.pop("cam_result", None)


def clear_upload(ss):
    ss.upload_bytes = None
    ss.upload_name = ""
    ss.pop("upload_widget", None)
    ss.pop("cam_result", None)
