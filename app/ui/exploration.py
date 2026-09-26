import numpy as np
import streamlit as st
from matplotlib import colormaps
from app.visualization import feature_maps
from app.ui.images import selected_image
from app.ui.controls import navigate

def render(ss):
    cfg, model = ss.settings, ss.model
    if not ss.trained:
        st.warning("Pesos não treinados: os mapas ainda não representam padrões aprendidos.")
    selected = selected_image(ss)
    if selected is not None:
        image, description = selected
        preview, maps_panel = st.columns([1, 3])
        with preview:
            st.subheader("Entrada")
            st.image(image[0].squeeze(), width="stretch", clamp=True)
            st.caption(description)
        with maps_panel:
            layer_name = f'conv_{cfg["layer_index"] + 1}'
            maps = feature_maps(model, image, layer_name)
            st.subheader(f"{layer_name} · {maps.shape[-1]} filtros")
            st.caption(f"Mapas {maps.shape[0]} × {maps.shape[1]}. Cores normalizadas por filtro para destacar padrões.")
            start = (cfg["filter_page"] - 1) * 16
            for row_start in range(start, min(start + 16, maps.shape[-1]), 4):
                columns = st.columns(4)
                for column, index in zip(columns, range(row_start, min(row_start + 4, maps.shape[-1]))):
                    plane = maps[:, :, index]
                    normalized = (plane - plane.min()) / max(float(np.ptp(plane)), 1e-8)
                    with column:
                        st.image(colormaps["viridis"](normalized)[:, :, :3],
                                 caption=f"Filtro {index + 1}", width="stretch")
        st.button("Continuar para Explicar", on_click=navigate, args=("Explicar",))
