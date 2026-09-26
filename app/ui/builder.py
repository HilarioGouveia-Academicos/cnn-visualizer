from io import BytesIO
import json
import streamlit as st
from app.datasets import DATASETS
from app.models import build_cnn
from app.state import architecture_config
from app.visualization import architecture_image
from app.ui.controls import control, navigate

@st.cache_data(max_entries=12, show_spinner=False)
def rendered_architecture(config_json, view, sizing, reversed_view, flat):
    config = json.loads(config_json)
    preview = build_cnn(
        num_conv=config["num_conv"], filters=config["filters"],
        kernel_size=(config["kernel"], config["kernel"]), activation=config["activation"],
        pooling=config["pooling"], dense_units=config["dense_units"],
        input_shape=DATASETS[config["dataset"]]["shape"],
    )
    drawing = architecture_image(preview, view, sizing, reversed_view, flat)
    buffer = BytesIO()
    drawing.save(buffer, format="PNG")
    return buffer.getvalue()



def render(ss):
    cfg, model = ss.settings, ss.model
    with st.container(horizontal=True):
        control("selectbox", "Representação", "view",
                options=["Camadas", "Grafo", "Functional View", "LeNet View"], width=200)
        control("selectbox", "Escala", "sizing",
                options=["Accurate", "Balanced", "Capped", "Logarithmic", "Relativa"], width=180,
                help="A escala altera apenas o desenho em Camadas, não a arquitetura nem seus cálculos.",
                disabled=cfg["view"] != "Camadas")
        with st.popover("Visualização avançada", icon=":material/settings:"):
            control("toggle", "Perspectiva invertida", "reversed_view", disabled=cfg["view"] != "Camadas")
            control("toggle", "Desenho plano", "flat_view", disabled=cfg["view"] != "Camadas")
            st.caption("A rede usa a API funcional do Keras. O grafo mostra suas conexões.")
    if cfg["view"] == "Functional View":
        st.caption("Blocos e conexões da rede atual. Esta CNN tem um único caminho sequencial.")
    elif cfg["view"] == "LeNet View":
        st.caption("Mapas de características empilhados da rede atual, no estilo LeNet.")
    with st.container(border=True):
        st.subheader("Arquitetura da rede")
        try:
            sizing = "relative" if cfg["sizing"] == "Relativa" else cfg["sizing"].lower()
            png = rendered_architecture(json.dumps(architecture_config(cfg), sort_keys=True),
                                        cfg["view"], sizing, cfg["reversed_view"], cfg["flat_view"])
            st.image(png, width="stretch")
            st.download_button("Baixar arquitetura", png, "arquitetura.png", "image/png",
                               icon=":material/download:")
        except Exception as exc:
            st.warning(f"Não foi possível desenhar a arquitetura: {exc}")
    a, b, c = st.columns(3)
    a.metric("Entrada", " × ".join(map(str, DATASETS[cfg["dataset"]]["shape"])))
    b.metric("Parâmetros", f"{model.count_params():,}".replace(",", "."))
    c.metric("Classes", "10")
    with st.expander("Dimensões e parâmetros por camada"):
        st.dataframe([{"Camada": layer.name, "Tipo": type(layer).__name__,
                       "Saída": str(tuple(layer.output.shape)), "Parâmetros": layer.count_params()}
                      for layer in model.layers], hide_index=True, width="stretch")
    st.info("Observe como o pooling reduz as dimensões espaciais e como Dense concentra os parâmetros.",
            icon=":material/school:")
    st.button("Continuar para Treinar", icon=":material/arrow_forward:", on_click=navigate, args=("Treinar",))
