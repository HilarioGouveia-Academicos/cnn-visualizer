# CNN Learn

Laboratório interativo, em português, para construir, treinar e explorar redes neurais convolucionais (CNNs). A interface usa **Streamlit**, os modelos são implementados com **TensorFlow/Keras** e a arquitetura é desenhada com **VisualKeras**.

**Aplicação online:** [CNN Learn](https://cnn-learn.streamlit.app/)

**Autor:** [Hilário Gouveia](https://www.linkedin.com/in/hilario-gouveia-2121062b4/)

## Funcionalidades

- Configuração de convoluções, filtros, kernel, ativação, pooling e camada densa, com validação das dimensões da rede.
- Visualização da arquitetura em Camadas, Grafo, Functional View (blocos e conexões) ou LeNet View (mapas empilhados), com exportação em PNG. Escala e perspectiva se aplicam ao modo Camadas.
- Treinamento em MNIST, FashionMNIST e CIFAR-10, com curvas de loss e acurácia de treino e validação atualizadas a cada época.
- Exploração de feature maps por camada convolucional, com até 16 filtros por página.
- Grad-CAM da classe prevista na última convolução, com controle de opacidade e exportação da sobreposição.
- Uso de imagens do conjunto de teste ou upload de PNG/JPEG para exploração e explicação.
- Salvamento local dos modelos e históricos por configuração, além de exportação das métricas em CSV.

## Instalação e execução

Tenha Python, pip e Git disponíveis. As dependências estão em [`requirements.txt`](requirements.txt). O VisualKeras usa uma revisão fixa do repositório oficial para incluir `functional_view` e `lenet_view`; as demais versões não estão fixadas.

Ao atualizar uma instalação existente, execute novamente `python -m pip install -r requirements.txt` e reinicie o Streamlit.

Na pasta do projeto, crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no terminal utilizado:

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Instale as dependências e inicie a aplicação a partir da raiz do projeto:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/main.py
```

Abra no navegador o endereço exibido pelo Streamlit, normalmente `http://localhost:8501`. Para encerrar, pressione `Ctrl+C` no terminal.

Os datasets são carregados sob demanda pelo Keras. O primeiro uso pode exigir conexão com a internet para download; abrir a etapa de construção não carrega o dataset.

## Como usar

1. **Construir:** escolha o dataset e configure a CNN. Consulte o desenho, as dimensões de saída e o número de parâmetros. Se uma combinação reduzir demais a imagem, diminua a quantidade de convoluções ou o tamanho do kernel.
2. **Treinar:** defina épocas e tamanho do lote (*batch size*) e clique em **Iniciar treinamento**. A aplicação reserva 15% do conjunto de treino para validação e mantém o conjunto de teste separado.
3. **Explorar:** selecione uma imagem de teste ou envie uma imagem, escolha a camada convolucional e navegue pelos filtros. É possível visualizar mapas antes do treino, mas eles ainda não representam padrões aprendidos.
4. **Explicar:** com o modelo treinado, selecione uma imagem e clique em **Gerar Grad-CAM**. Compare a imagem original, o mapa de calor e a sobreposição, e consulte as probabilidades por classe.
5. **Comparar**, permite selecionar de duas a seis execuções do mesmo dataset, incluindo treinos não salvos da sessão. Exibe uma tabela de parâmetros, barras de acurácia/loss finais de validação e curvas de treino e validação por época. A tabela pode ser baixada em CSV.

A melhor acurácia é identificada separadamente da acurácia final; os pesos salvos são os da última época. Divisões ou quantidades de imagens diferentes geram um aviso. A comparação não carrega os pesos e preserva o modelo atual. Checkpoints antigos sem registro de execução não participam desta primeira versão.


A configuração inicial usa MNIST, duas convoluções com 32 filtros e kernel 3 × 3, ativação ReLU, MaxPooling, 64 neurônios na camada densa, três épocas e lotes de 64 imagens.

## Datasets e imagens

| Dataset | Entrada | Classes |
| --- | --- | --- |
| MNIST | 28 × 28, escala de cinza | 10 dígitos |
| FashionMNIST | 28 × 28, escala de cinza | 10 categorias de roupas e acessórios |
| CIFAR-10 | 32 × 32, RGB | 10 categorias de objetos e animais |

Os pixels são normalizados para o intervalo de 0 a 1. Imagens enviadas são redimensionadas e convertidas para os canais esperados pelo dataset selecionado. O upload serve para explorar e explicar previsões; o treinamento utiliza os datasets listados acima.

Grad-CAM destaca regiões que contribuem para a classe prevista. Não é uma segmentação nem uma prova de causalidade.

## Estado e modelos salvos

Ao abrir a aplicação ou selecionar outro dataset, o treino salvo mais recente desse dataset é recuperado automaticamente, incluindo arquitetura, pesos e métricas. Se não houver um treino válido, a aplicação cria um modelo não treinado. Arquivos inválidos geram um aviso e a aplicação tenta um treino anterior.

Navegar entre as etapas preserva os parâmetros e o modelo da sessão. Alterar manualmente a arquitetura cria um novo modelo e invalida o treinamento atual. Cada clique em **Iniciar treinamento** começa com pesos novos.

Após o treinamento, a execução fica **não salva**, disponível apenas na sessão. Na etapa **Treinar**, selecione um resultado em **Execuções não salvas**, informe um nome e clique em **Salvar execução**. Trocar o dataset ou iniciar outro treino mantém os resultados anteriores nessa lista até o encerramento da sessão.

Cada execução escolhida recebe um identificador único e é armazenada em `saved_models/runs/<id>/`:

- `model.keras`: arquitetura e pesos da última época.
- `metadata.json`: nome, data UTC, arquitetura, épocas solicitadas e concluídas, batch size, contagens de imagens, divisão de validação, duração incluindo preparação dos dados, histórico e métricas finais/melhores.

Repetições com parâmetros iguais são execuções distintas e não sobrescrevem treinos salvos. O campo **Semente aleatória**, na etapa **Treinar**, tem padrão 42 e aceita inteiros de 0 a 4294967295. A semente é aplicada a Python, NumPy e TensorFlow antes da criação do modelo e fica registrada com a execução. Os parâmetros registrados refletem o momento do treino, mesmo que os controles sejam alterados antes de salvar. Execuções antigas podem ter semente não definida (`null`).

A semente ajuda a repetir a inicialização e o embaralhamento, mas não garante resultados idênticos entre equipamentos, versões de bibliotecas ou treinamentos concorrentes no mesmo servidor. Operações determinísticas não são forçadas.

Use **Treinos salvos** e **Carregar treino salvo** para recuperar uma execução do dataset selecionado. Os checkpoints anteriores no formato `<identificador>.keras` e `<identificador>.json` continuam compatíveis com a restauração automática. Os arquivos legados `cnn_model.h5` e `metrics.json` não são carregados por esse fluxo.

O botão salva no computador que executa o servidor. No Community Cloud, isso não representa armazenamento durável nem download para o computador do visitante.

## Estrutura do projeto

```text
app/
  main.py             Inicialização e seleção da etapa
  ui/
    navigation.py     Dock e sidebar
    builder.py        Etapa Construir
    training.py       Etapa Treinar e gráficos
    exploration.py    Feature maps
    explanation.py    Grad-CAM
    controls.py       Integração dos widgets com o estado
    images.py         Seleção e preparação da imagem
    data.py           Cache de datasets da interface
  training.py         Execução do treinamento e progresso por callback
  persistence.py      Carregamento e salvamento dos modelos
  settings.py         Configurações da instalação
  datasets.py         Metadados, carregamento e normalização dos datasets
  models.py           Construção e validação da CNN
  state.py            Configurações e transições de estado da sessão
  visualization.py    Desenho da rede, feature maps e Grad-CAM
assets/               Estilos e recursos visuais
notebooks/            Notebook de exploração
saved_models/         Modelos e métricas persistidos localmente
tests/                Testes de modelos, visualizações e interface
requirements.txt      Dependências Python
```

## Treinamento no Streamlit Community Cloud

Para bloquear novos treinamentos, adicione em **Settings → Secrets** da aplicação:

```toml
[app]
allow_training = false
```

O botão e os parâmetros de treinamento ficam ocultos, e a execução também é bloqueada no servidor. Modelos salvos, métricas, feature maps e Grad-CAM continuam disponíveis. Disponibilize os arquivos `.keras` e `.json` correspondentes em `saved_models/` no deploy.

Sem essa configuração, o treinamento permanece habilitado. Para testar localmente, use o mesmo conteúdo em `.streamlit/secrets.toml`. Para reativar, defina `allow_training = true`.

## Testes

Com o ambiente virtual ativado, execute na raiz do projeto:

```bash
python -m pytest tests/ -q
```

O teste de treinamento em `tests/test_models.py` usa MNIST e pode precisar baixá-lo. Para executar apenas os testes de interface, visualização e treinamento com dados sintéticos, sem download de datasets:

```bash
python -m pytest tests/test_workspace.py -q
```
