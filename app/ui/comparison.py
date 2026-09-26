"""Compare execution metadata without loading models or datasets."""
import pandas as pd
import streamlit as st
from app.persistence import saved_executions


def available_records(ss, dataset):
    records = {record["run_id"]: dict(record, source="Salva")
               for _, record in saved_executions(dataset)}
    for run_id, run in ss.session_runs.items():
        record = run["record"]
        if not run["saved"] and record["config"]["dataset"] == dataset:
            records[run_id] = dict(record, source="Não salva", name="Execução da sessão")
    return records


def label(record):
    return f'{record["name"]} · {record["run_id"][:8]} · {record["source"]}'


def summary(records):
    rows = []
    for record in records:
        cfg, train, history = record["config"], record.get("training", {}), record["history"]
        rows.append({
            "Execução": label(record), "Data UTC": record["created_at"],
            "Convoluções": cfg["num_conv"], "Filtros": cfg["filters"],
            "Kernel": cfg["kernel"], "Ativação": cfg["activation"],
            "Pooling": cfg["pooling"], "Dense": cfg["dense_units"],
            "Épocas": len(history["loss"]), "Batch size": train.get("batch_size"),
            "Semente": train.get("seed"), "Imagens de treino": train.get("training_images"),
            "Imagens de validação": train.get("validation_images"),
            "Acurácia final de validação (%)": 100 * history["val_accuracy"][-1],
            "Melhor acurácia de validação (%)": 100 * max(history["val_accuracy"]),
            "Loss final de validação": history["val_loss"][-1],
            "Duração (s)": record.get("duration_seconds"),
        })
    return pd.DataFrame(rows)


def evaluation_signature(record):
    train = record.get("training", {})
    return tuple(train.get(key) for key in
                 ("total_images", "validation_images", "validation_split", "split_method"))


def render(ss):
    records = available_records(ss, ss.comparison_dataset)
    if not records:
        st.info("Nenhuma execução registrada para este dataset. Faça um treinamento ou salve uma execução na etapa Treinar.")
        st.caption("Checkpoints antigos sem registro de execução ainda podem ser carregados em Treinar, mas não entram nesta comparação.")
        return
    selected = st.multiselect("Execuções para comparar", options=list(records),
                              default=list(records)[:2], max_selections=6,
                              format_func=lambda key: label(records[key]),
                              key="compare_runs_" + ss.comparison_dataset)
    if len(selected) < 2:
        st.info("Selecione pelo menos duas execuções do mesmo dataset para comparar.")
        return
    chosen = [records[key] for key in selected]
    signatures = [evaluation_signature(record) for record in chosen]
    if len(set(signatures)) > 1 or any(None in signature for signature in signatures):
        st.warning("As divisões ou quantidades de imagens diferem, ou não estão completamente registradas. Os resultados podem não ser diretamente comparáveis.")
    st.caption("Métricas de validação, não de teste. Mesmos tamanhos de divisão não comprovam imagens idênticas; os registros atuais não armazenam seus identificadores. Os modelos usam os pesos da última época.")
    frame = summary(chosen)
    st.subheader("Parâmetros e resultados")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button("Baixar comparação CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                       "comparacao.csv", "text/csv")
    st.subheader("Resultado final")
    indexed = frame.set_index("Execução")
    accuracy, loss = st.columns(2)
    with accuracy:
        st.caption("Acurácia final de validação (%) — maior é melhor")
        st.bar_chart(indexed[["Acurácia final de validação (%)"]])
    with loss:
        st.caption("Loss final de validação — menor é melhor")
        st.bar_chart(indexed[["Loss final de validação"]])
    st.subheader("Evolução por época")
    metric = st.selectbox("Métrica", ["Acurácia", "Loss"], key="comparison_metric")
    key = "accuracy" if metric == "Acurácia" else "loss"
    for title, prefix in (("Validação", "val_"), ("Treino", "")):
        st.caption(title)
        curves = pd.DataFrame({label(record): pd.Series(record["history"][prefix + key],
                                   index=range(1, len(record["history"][prefix + key]) + 1))
                               for record in chosen})
        curves.index.name = "Época"
        st.line_chart(curves)
    st.caption("As curvas terminam na última época de cada execução; não há extrapolação. A duração registrada inclui preparação dos dados.")
