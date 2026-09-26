import pandas as pd
import streamlit as st
from time import perf_counter
from copy import deepcopy
from app.settings import training_allowed
from app.persistence import load_training, execution_record, save_execution, load_execution
from app.state import accept_training, register_execution
from app.training import train_model
from app.ui.data import cached_dataset
from app.ui.controls import navigate

def plot_history(history, slot):
    with slot.container():
        loss, accuracy = st.columns(2)
        for column, title, metric in [(loss, "Loss", "loss"), (accuracy, "Acurácia", "accuracy")]:
            with column:
                st.subheader(title)
                if history:
                    frame = pd.DataFrame({
                        "Treino": history[metric], "Validação": history["val_" + metric],
                    }, index=pd.Index(range(1, len(history[metric]) + 1), name="Época"))
                    st.line_chart(frame, color=["#4466dd", "#259d86"], height=280)
                else:
                    st.caption("As curvas aparecerão após a primeira época.")



def render(ss, allow_training, train_clicked, load_clicked, status_slot):
    cfg = ss.settings
    if not allow_training and not ss.trained:
        st.info("Não há modelo treinado carregado para esta configuração. Selecione um dataset com treino salvo ou carregue um modelo compatível.")
    elif ss.ever_trained and not ss.trained:
        st.info("A configuração mudou. Inicie um novo treinamento para esta arquitetura.")
    if allow_training:
        st.caption("Um novo treinamento começa com pesos novos. Navegar entre etapas preserva o modelo atual.")
    progress_slot = st.empty()
    chart_slot = st.empty()
    if load_clicked:
        try:
            if ss.get("selected_saved_run"):
                loaded, record = load_execution(ss.selected_saved_run)
                cfg.update(record["config"])
                history = record["history"]
            else:
                loaded, history = load_training(cfg)
            accept_training(ss, loaded, history)
            st.rerun()
        except Exception as exc:
            st.error(f"Não foi possível carregar o treinamento: {exc}")
    if train_clicked and training_allowed():
        def on_epoch(epoch, history):
            progress_slot.progress((epoch + 1) / cfg["epochs"], text=f'Época {epoch + 1} de {cfg["epochs"]}')
            plot_history(history, chart_slot)

        status_slot.badge("Treinando", color="blue")
        try:
            run_config = deepcopy(cfg)
            counts = {}
            def dataset_for_run(name):
                data = cached_dataset(name)
                counts["total"] = len(data[0])
                return data
            started = perf_counter()
            progress_slot.progress(0, text="Preparando a primeira época…")
            candidate, history = train_model(run_config, dataset_for_run, on_epoch)
            record = execution_record(run_config, history, perf_counter() - started, counts["total"])
            accept_training(ss, candidate, history)
            register_execution(ss, candidate, record)
            status_slot.badge("Treinado", color="green")
            st.success("Treinamento concluído. Explore os filtros ou gere uma explicação.")
        except Exception as exc:
            status_slot.badge("Treinado" if ss.trained else "Não treinado",
                              color="green" if ss.trained else "gray")
            progress_slot.empty()
            st.error(f"Não foi possível concluir o treinamento: {exc}")
    plot_history(ss.history, chart_slot)
    if ss.history:
        a, b = st.columns(2)
        a.metric("Acurácia de treino", f'{ss.history["accuracy"][-1]:.1%}')
        b.metric("Acurácia de validação", f'{ss.history["val_accuracy"][-1]:.1%}')
        st.download_button("Baixar métricas", pd.DataFrame(ss.history).to_csv(index_label="Época"),
                           "metricas.csv", "text/csv", icon=":material/download:")
        st.button("Explorar feature maps", on_click=navigate, args=("Explorar",))
    render_session_runs(ss)


def render_session_runs(ss):
    pending = {key: value for key, value in ss.session_runs.items() if not value["saved"]}
    if not pending:
        return
    st.subheader("Execuções não salvas")
    st.warning("Estes resultados existem apenas nesta sessão. Salve os que deseja manter antes de fechar ou reiniciar a aplicação.")
    selected = st.selectbox("Execução para salvar", options=list(pending), key="pending_run",
                            format_func=lambda key: (
                                f'{pending[key]["record"]["config"]["dataset"]} · '
                                f'{pending[key]["record"]["created_at"]} · '
                                f'validação {pending[key]["record"]["metrics"]["final_val_accuracy"]:.1%}'
                            ))
    run = pending[selected]
    record = run["record"]
    with st.expander("Parâmetros desta execução"):
        st.json({"arquitetura": record["config"], "treinamento": record["training"],
                 "duracao_segundos": record["duration_seconds"], "metricas": record["metrics"]})
    name = st.text_input("Nome da execução", value=f'{record["config"]["dataset"]} — {selected[:8]}',
                         key="run_name_" + selected)
    if st.button("Salvar execução", key="save_execution", type="primary"):
        try:
            run["record"] = save_execution(record, run["model"], name)
            run["saved"] = True
            # Persisted runs no longer need to keep a second model reference.
            run.pop("model", None)
            st.success("Execução salva. Ela está disponível em Treinos salvos na barra lateral.")
        except Exception as exc:
            st.error(f"Não foi possível salvar a execução: {exc}")
