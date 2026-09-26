from pathlib import Path
import streamlit as st
from app.datasets import DATASETS
from app.persistence import checkpoint_path, saved_executions
from app.ui.controls import control, navigate, keep_upload, clear_upload

ROOT = Path(__file__).resolve().parents[2]
STEPS = {
    "Construir": ("account_tree", "Monte sua rede e acompanhe como cada escolha altera a arquitetura."),
    "Treinar": ("play_circle", "Acompanhe como a rede aprende e compare treino e validação."),
    "Explorar": ("grid_view", "Observe as respostas dos filtros em cada camada convolucional."),
    "Explicar": ("lightbulb", "Descubra quais regiões contribuem para a classificação."),
    "Comparar": ("compare_arrows", "Compare parâmetros e resultados das execuções."),
}

def render_navigation(ss, allow_training):
    cfg = ss.settings
    st.html((ROOT / "assets" / "workspace.css").read_text(encoding="utf-8"))
    with st.container(key="activity_dock"):
        st.markdown("**CNN**")
        for name, (icon, _) in STEPS.items():
            st.button(name, icon=f":material/{icon}:", key="nav_" + name,
                      type="primary" if ss.stage == name else "secondary",
                      width="stretch", on_click=navigate, args=(name,))
        st.caption("VISUALIZER")

    stage = ss.stage
    train_clicked = load_clicked = cam_clicked = False
    with st.sidebar:
        st.caption("PARÂMETROS")
        st.header(stage)
        pending_count = sum(not run["saved"] for run in ss.session_runs.values())
        if pending_count:
            st.warning(f"{pending_count} execução(ões) não salva(s). Disponíveis em Treinar até o fim desta sessão.")
        if stage == "Comparar":
            st.selectbox("Dataset para comparar", options=list(DATASETS), key="comparison_dataset")
            st.caption("Selecione as execuções no painel principal. O modelo atual será preservado.")
        elif stage == "Construir":
            control("selectbox", "Dataset", "dataset", options=list(DATASETS))
            with st.expander("Convoluções", expanded=True):
                control("slider", "Camadas convolucionais", "num_conv", min_value=1, max_value=5)
                control("select_slider", "Filtros por camada", "filters", options=[16, 32, 64, 96, 128])
                control("selectbox", "Kernel", "kernel", options=[3, 5], format_func=lambda n: f"{n} × {n}")
                control("selectbox", "Ativação", "activation", options=["relu", "sigmoid", "tanh"])
            with st.expander("Pooling e classificação", expanded=False):
                control("selectbox", "Pooling", "pooling", options=["MaxPooling2D", "AveragePooling2D"])
                control("slider", "Neurônios Dense", "dense_units", min_value=32, max_value=256, step=32)
            st.caption("Cada bloco aplica convolução e pooling 2 × 2. As alterações aparecem no painel.")
        else:
            st.caption(f'{cfg["dataset"]} · {cfg["num_conv"]} convoluções · {cfg["filters"]} filtros')
            st.button("Editar arquitetura", icon=":material/tune:", on_click=navigate, args=("Construir",))
            if stage == "Treinar":
                if allow_training:
                    control("number_input", "Épocas", "epochs", min_value=1, max_value=100, step=1)
                    control("selectbox", "Batch size", "batch_size", options=[16, 32, 64, 128, 256])
                    control("number_input", "Semente aleatória", "seed", min_value=0,
                            max_value=4294967295, step=1,
                            help="Controla a inicialização dos pesos e o embaralhamento. Use a mesma semente para repetir um experimento no mesmo ambiente.")
                    st.caption("15% do conjunto de treino é reservado para validação. O teste permanece separado.")
                    train_clicked = st.button("Iniciar treinamento", icon=":material/play_arrow:",
                                              type="primary", width="stretch")
                else:
                    st.info("Treinamento desabilitado nesta instalação. Use os modelos salvos para explorar e explicar previsões.")
                checkpoint = checkpoint_path(cfg)
                records = saved_executions(cfg["dataset"])
                ss.selected_saved_run = None
                if records:
                    options = {str(path): record for path, record in records}
                    ss.selected_saved_run = st.selectbox(
                        "Treinos salvos", options=list(options), key="saved_run_selector",
                        format_func=lambda path: f'{options[path]["name"]} · {options[path]["created_at"]} · {options[path]["run_id"][:8]}')
                if records or (checkpoint.exists() and checkpoint.with_suffix(".json").exists()):
                    load_clicked = st.button("Carregar treino salvo", icon=":material/folder_open:", width="stretch")
            else:
                control("selectbox", "Origem da imagem", "image_source", options=["Dataset", "Upload"])
                if cfg["image_source"] == "Dataset":
                    control("number_input", "Índice da imagem de teste", "image_index",
                            min_value=0, max_value=9999, step=1)
                else:
                    st.file_uploader("Imagem PNG ou JPEG", type=["png", "jpg", "jpeg"],
                                     key="upload_widget", on_change=keep_upload)
                    if ss.upload_bytes:
                        st.caption(f"Selecionada: {ss.upload_name}")
                        st.button("Remover imagem", on_click=clear_upload)
                if stage == "Explorar":
                    cfg["layer_index"] = min(cfg["layer_index"], cfg["num_conv"] - 1)
                    control("selectbox", "Camada convolucional", "layer_index",
                            options=list(range(cfg["num_conv"])), format_func=lambda i: f"conv_{i + 1}")
                    max_page = (cfg["filters"] + 15) // 16
                    cfg["filter_page"] = min(cfg["filter_page"], max_page)
                    control("number_input", "Página de filtros", "filter_page", min_value=1, max_value=max_page)
                    st.caption("Até 16 filtros por página.")
                else:
                    control("slider", "Opacidade do heatmap", "opacity", min_value=0.0, max_value=1.0, step=0.05)
                    st.caption(f'Grad-CAM na última convolução: conv_{cfg["num_conv"]}.')
                    cam_clicked = st.button("Gerar Grad-CAM", type="primary", width="stretch",
                                            disabled=not ss.trained)
        with st.expander("Como usar"):
            st.markdown("**1. Construir:** configure sua CNN.\n\n**2. Treinar:** acompanhe o aprendizado."
                        "\n\n**3. Explorar:** examine os filtros.\n\n**4. Explicar:** interprete a previsão."
                        "\n\n**5. Comparar:** compare os resultados das execuções.")


    return train_clicked, load_clicked, cam_clicked
