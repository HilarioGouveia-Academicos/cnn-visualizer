from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from app.datasets import DATASETS
from app.ui.data import cached_dataset

def selected_image(ss):
    cfg = ss.settings
    shape = DATASETS[cfg["dataset"]]["shape"]
    if cfg["image_source"] == "Upload":
        if not ss.upload_bytes:
            st.info("Escolha uma imagem na barra de parâmetros.")
            return None
        try:
            source = ImageOps.exif_transpose(Image.open(BytesIO(ss.upload_bytes)))
            source = source.convert("L" if shape[-1] == 1 else "RGB")
            source = source.resize((shape[1], shape[0]))
            image = np.asarray(source, dtype=np.float32).reshape(shape) / 255.0
            return image[None, ...], ss.upload_name
        except (ValueError, OSError) as exc:
            st.error(f"Não foi possível abrir a imagem: {exc}")
            return None
    try:
        _, _, images, labels = cached_dataset(cfg["dataset"])
    except Exception as exc:
        st.error(f"Não foi possível carregar o dataset. Verifique sua conexão e tente novamente. {exc}")
        return None
    index = min(cfg["image_index"], len(images) - 1)
    label = DATASETS[cfg["dataset"]]["classes"][int(labels[index])]
    return images[index:index + 1], f"Imagem {index} · Classe real: {label}"
