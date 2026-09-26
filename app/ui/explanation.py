from io import BytesIO
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib import colormaps
from app.datasets import DATASETS
from app.visualization import grad_cam
from app.ui.images import selected_image
from app.ui.controls import navigate

def render(ss, allow_training, cam_clicked):
    cfg, model = ss.settings, ss.model
    if not ss.trained:
        st.info("Treine o modelo antes de explicar uma classificação." if allow_training else
                "Selecione um dataset com treino salvo para explicar uma classificação. O treinamento está desabilitado nesta instalação.",
                icon=":material/lightbulb:")
        st.button("Ir para Treinar", on_click=navigate, args=("Treinar",))
    else:
        selected = selected_image(ss)
        if selected is not None:
            image, description = selected
            probabilities = np.asarray(model(image, training=False))[0]
            predicted = int(probabilities.argmax())
            classes = DATASETS[cfg["dataset"]]["classes"]
            st.subheader(f"Previsão: {classes[predicted]}")
            st.caption(f"{description} · Probabilidade prevista: {probabilities[predicted]:.1%}")
            image_key = hashlib.sha256(image.tobytes()).hexdigest()
            if cam_clicked:
                with st.spinner("Calculando Grad-CAM…"):
                    heatmap, _ = grad_cam(model, image, f'conv_{cfg["num_conv"]}', predicted)
                    ss.cam_result = (image_key, heatmap)
            original, heat, overlay_panel = st.columns(3)
            original.image(image[0].squeeze(), caption="Imagem original", width="stretch")
            if "cam_result" in ss and ss.cam_result[0] == image_key:
                heatmap = ss.cam_result[1]
                if not np.any(heatmap):
                    st.info("Não houve contribuição positiva nesta camada para a classe prevista.")
                resized = np.asarray(Image.fromarray(heatmap).resize(
                    (image.shape[2], image.shape[1]), Image.Resampling.BILINEAR))
                colored = colormaps["inferno"](np.clip(resized, 0, 1))[:, :, :3].astype(np.float32)
                base = np.repeat(image[0], 3, axis=-1) if image.shape[-1] == 1 else image[0]
                overlay = np.clip((1 - cfg["opacity"]) * base + cfg["opacity"] * colored, 0, 1)
                heat.image(colored, caption="Grad-CAM · escuro = menor contribuição", width="stretch")
                overlay_panel.image(overlay, caption="Sobreposição", width="stretch")
                buffer = BytesIO()
                Image.fromarray((overlay * 255).astype(np.uint8)).save(buffer, format="PNG")
                st.download_button("Baixar sobreposição", buffer.getvalue(), "grad-cam.png", "image/png")
            else:
                heat.info("Use Gerar Grad-CAM na barra de parâmetros.")
            with st.expander("Probabilidades por classe"):
                st.bar_chart(pd.DataFrame({"Probabilidade": probabilities}, index=classes))
            st.caption("Grad-CAM destaca contribuições para a classe prevista; não é uma segmentação nem prova de causalidade.")
