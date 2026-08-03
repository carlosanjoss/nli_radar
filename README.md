# Classificadores NLI one-shot — camadas 3 e 4

Este projeto executa separadamente:

- a camada 3 de intensidade/toxicidade, preservando a lógica do código informado;
- a camada 4 de dimensões discursivas, construída a partir das hipóteses do manual;
- a verificação descritiva das duas saídas sobre o mesmo CSV.

## Integração Gemma → NLI → PostgreSQL

O fluxo de produção lê diretamente o mesmo PostgreSQL usado por
`../gemma_hate_inference`. Para cada post ou comentário, ele considera somente a
avaliação Gemma bem-sucedida mais recente. O conteúdo entra no NLI apenas quando
essa avaliação possui `is_hate = true`.

As camadas são processadas e persistidas separadamente:

| Camada | Resultado final | Scores das hipóteses |
|---|---|---|
| 3 | `radar_nli_layer3_results` | `radar_nli_layer3_hypothesis_scores` |
| 4 | `radar_nli_layer4_results` | `radar_nli_layer4_hypothesis_scores` |

As execuções das duas camadas são auditadas em `radar_nli_runs`. As views
`v_radar_nli_layer3_results` e `v_radar_nli_layer4_results` acrescentam o modelo,
prompt, probabilidade e data da classificação Gemma que liberou cada conteúdo.

Configure `POSTGRES_URI` no ambiente ou copie `.env.example` para `.env`. Quando a
variável não estiver no projeto, o programa também procura os arquivos `.env` de
`../gemma_hate_inference` e `../social_data_pipeline`.

Crie as tabelas sem executar inferência:

```powershell
python run_postgres.py --migrate-only
```

Teste as duas camadas com dez conteúdos hate pendentes por camada:

```powershell
python run_postgres.py --limit 10
```

Depois, processe todos os posts e comentários hate ainda pendentes:

```powershell
python run_postgres.py
```

Também é possível executar uma camada isoladamente ou filtrar a origem:

```powershell
python run_postgres.py --layer layer3
python run_postgres.py --layer layer4
python run_postgres.py --platform reddit --platform youtube
python run_postgres.py --content-kind post
python run_postgres.py --content-kind comment
python run_postgres.py --gemma-model gemma4:e4b `
  --gemma-prompt-version gemma4b_oneshot_direto_identitario
```

O fluxo é idempotente por camada. Para cada plataforma, somente um conteúdo
canônico por texto entra na inferência. Um resultado existente com o mesmo texto,
modelo NLI, arquivo de hipóteses e configuração não é executado novamente. Se o
Gemma avaliar novos textos como hate, basta rodar o comando outra vez; somente os
pendentes serão selecionados. Os CSVs em `outputs/postgres` são cópias de auditoria,
enquanto a saída principal fica no PostgreSQL.

No modo PostgreSQL, cada lote é salvo em uma transação antes de o NLI avançar para
o lote seguinte. A mesma transação grava o resultado final, os scores das hipóteses
e incrementa `radar_nli_runs.processed_count`. Se houver interrupção, os lotes já
confirmados permanecem no banco e a próxima execução seleciona somente os pendentes.

Consulte o progresso sem carregar o modelo:

```powershell
python run_postgres.py --status
python run_postgres.py --status --layer layer3
python run_postgres.py --status --run-id 4
```

O checkpoint configurado é
[`joeddav/xlm-roberta-large-xnli`](https://huggingface.co/joeddav/xlm-roberta-large-xnli).
Ele é um modelo XLM-RoBERTa Large ajustado para NLI multilíngue e classificação
zero-shot. O próprio model card documenta a ordem dos logits como contradição,
neutralidade e entailment e recomenda, no caso multirrótulo, normalizar entailment
contra contradiction.

Leia também `ANALISE_DOCUMENTOS.md`. Esse arquivo registra as inconsistências
encontradas nos documentos e no código anterior e explica todas as decisões de
implementação.

## 1. Estrutura da pasta

```text
radar_nli_oneshot/
├── ANALISE_DOCUMENTOS.md
├── README.md
├── requirements.txt
├── run_all.py
├── common/
│   ├── aggregation.py
│   ├── config.py
│   ├── data.py
│   ├── hypotheses.py
│   ├── nli.py
│   ├── reporting.py
│   └── runner.py
├── data/
│   └── reddit_tupye_2500_psicologia_sem_gabarito.csv  # adicionar aqui
├── layer3/
│   ├── config.yaml
│   ├── hypotheses.yaml
│   ├── postprocessing.py
│   └── run.py
├── layer4/
│   ├── config.yaml
│   ├── hypotheses.yaml
│   └── run.py
├── evaluation/
│   ├── compare_layers.py
│   └── evaluate_with_gold.py
├── outputs/
└── tests/
```

As camadas não compartilham hipóteses, thresholds nem regras finais. Elas
compartilham apenas o carregamento do modelo, a leitura do CSV e a infraestrutura de
inferência. Isso evita a mistura conceitual que existia nos nomes `layer4_*` do
código apresentado como camada 3.

## 2. Preparar o ambiente

### Windows PowerShell

Abra o PowerShell dentro da pasta `radar_nli_oneshot` e execute:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se a política do PowerShell bloquear a ativação do ambiente, execute apenas na
sessão atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch com CUDA

O comando genérico do `requirements.txt` pode instalar uma distribuição que não
corresponda à sua versão de CUDA. Se você possui GPU NVIDIA, instale o PyTorch com o
comando indicado para seu sistema em <https://pytorch.org/get-started/locally/> e
depois instale o restante das dependências.

Confirme o ambiente:

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

O modelo possui aproximadamente 0,6 bilhão de parâmetros. Se ocorrer falta de
memória na GPU, reduza `model.batch_size` nos dois arquivos `config.yaml` de 8 para
4, 2 ou 1. Em CPU o processo funciona, mas a amostra completa exigirá muito mais
tempo.

## 3. Colocar e conferir o CSV

Copie o arquivo para:

```text
data/reddit_tupye_2500_psicologia_sem_gabarito.csv
```

O programa tenta detectar automaticamente a coluna de texto nesta ordem:
`cleaned_body`, `body`, `text`, `texto`, `comment`, `comentario`, `comentário` e
`content`.

Ele tenta detectar o identificador entre `clean_id`, `comment_id`, `id`,
`external_id` e `row_id`. Se não encontrar, usa o índice original da linha.

Antes da inferência completa, confira as colunas:

```bash
python -c "import pandas as pd; p='data/reddit_tupye_2500_psicologia_sem_gabarito.csv'; d=pd.read_csv(p, sep=None, engine='python'); print(d.shape); print(d.columns.tolist()); print(d.head(2).to_string())"
```

Se a coluna de texto tiver outro nome, informe-a explicitamente em todos os comandos,
por exemplo:

```bash
python -m layer4.run --text-column texto_comentario --limit 10
```

## 4. Primeiro teste: somente dez comentários

Não comece pelas 2.500 linhas. Faça um teste que valide modelo, GPU, coluna de texto,
hipóteses e escrita das saídas:

```bash
python -m layer3.run --limit 10 --overwrite
python -m layer4.run --limit 10 --overwrite
python -m evaluation.compare_layers \
  --layer3 outputs/layer3/layer3_predictions.csv \
  --layer4 outputs/layer4/layer4_predictions.csv \
  --output-dir outputs/evaluation
```

No PowerShell, o mesmo comando de comparação pode ser escrito em uma linha:

```powershell
python -m evaluation.compare_layers --layer3 outputs/layer3/layer3_predictions.csv --layer4 outputs/layer4/layer4_predictions.csv --output-dir outputs/evaluation
```

Abra os três arquivos abaixo e confira se os textos e identificadores são os
esperados:

```text
outputs/layer3/layer3_predictions.csv
outputs/layer4/layer4_predictions.csv
outputs/evaluation/cross_layer_verification.csv
```

Somente depois dessa inspeção execute a amostra completa.

## 5. Executar somente a camada 3

```bash
python -m layer3.run --overwrite
```

Para informar caminhos e colunas:

```bash
python -m layer3.run \
  --input data/reddit_tupye_2500_psicologia_sem_gabarito.csv \
  --text-column cleaned_body \
  --id-column comment_id \
  --output-dir outputs/layer3 \
  --overwrite
```

A camada 3 aplica, nesta ordem:

1. inferência NLI para as oito dimensões;
2. agregação das hipóteses positivas;
3. desconto das hipóteses negativas e neutras;
4. boosts lexicais para ameaças, violência sexual e desumanização;
5. bloqueio de falsos positivos em discussão jurídica, acadêmica ou metalinguística;
6. redução contextual em citações e discussões sobre o próprio fenômeno;
7. cálculo do score final de toxicidade;
8. atribuição de `nao_toxico`, `baixa`, `media` ou `alta`.

A fórmula preservada é:

\[
T = 0{,}45S + 0{,}25I + 0{,}20C + 0{,}10\bar{x},
\]

em que `S` é o maior score severo, `I` o maior score identitário, `C` o maior
score de incivilidade e `x̄` a média das oito dimensões.

## 6. Executar somente a camada 4

```bash
python -m layer4.run --overwrite
```

A camada 4 avalia 63 pares premissa-hipótese por comentário, distribuídos entre
cinco dimensões e doze construtos. Para 2.500 comentários, isso corresponde a
157.500 pares NLI.

Ela é multirrótulo. Um comentário pode ativar mais de uma dimensão. A coluna
`primary_dimension` serve apenas para ordenar e inspecionar; ela não substitui as
colunas booleanas de cada dimensão nem `active_dimensions`.

## 7. Executar as duas camadas com um único comando

```bash
python run_all.py --overwrite
```

O modelo é carregado uma única vez e reutilizado pelas duas camadas. O comando
executa a camada 3, a camada 4 e a verificação cruzada.

Para um teste pequeno:

```bash
python run_all.py --limit 10 --overwrite
```

Para uma coluna de texto específica:

```bash
python run_all.py --text-column cleaned_body --id-column comment_id --overwrite
```

## 8. Aplicar o gate da camada 1

Segundo o manual, apenas textos já classificados como discurso de ódio devem chegar
às camadas seguintes. Se o CSV contiver uma coluna como `pred_hate`, execute:

```bash
python run_all.py --gate-column pred_hate --overwrite
```

São aceitos como positivos, por padrão: `1`, `true`, `hate`, `sim` e `positivo`.
Esses valores podem ser alterados em `input.gate.positive_values` nos YAMLs.

Se o CSV não possuir previsão da camada 1, não invente um gate. Execute toda a
amostra, mas interprete os resultados como análise exploratória. Uma dimensão da
camada 4 detectada em texto não odioso não transforma automaticamente esse texto em
discurso de ódio.

## 9. Como cada score NLI é calculado

Para cada comentário, o texto é a premissa e cada frase do YAML é a hipótese. O
tokenizador recebe o par e trunca somente a premissa quando o total ultrapassa 512
tokens, preservando a hipótese inteira.

O modelo devolve logits de contradição, neutralidade e entailment. O pacote grava o
softmax completo para auditoria, mas calcula a probabilidade usada na classificação
apenas entre contradição e entailment:

\[
p_h = \frac{e^{z_E}}{e^{z_E}+e^{z_C}}.
\]

Isso é importante porque as dimensões são independentes. Não há obrigação de os
scores das cinco dimensões somarem 1.

Dentro de um construto, o score positivo bruto é:

\[
s^+ = 0{,}70\max(p_1,\ldots,p_k)+0{,}30\operatorname{média}(p_1,\ldots,p_k).
\]

As hipóteses `NEG` e `NEU` descrevem fronteiras conceituais e funcionam como
contraprovas:

\[
s_c = \max\left(0, s^+ - 0{,}50\max(p_{NEG},p_{NEU})\right).
\]

O score da dimensão aplica novamente 70% do maior construto e 30% da média dos
construtos. Todos esses pesos podem ser alterados nos YAMLs e aparecem explicitamente
para permitir reprodução do experimento.

## 10. Arquivos produzidos

Cada camada gera quatro arquivos principais:

| Arquivo | Conteúdo |
|---|---|
| `*_predictions.csv` | uma linha por comentário, scores agregados, flags, dimensão principal e texto |
| `*_hypothesis_scores.csv` | uma linha por comentário e hipótese, com as quatro probabilidades registradas |
| `*_dimension_summary.csv` | contagem ativa, porcentagem, média, mediana, percentil 95 e máximo por dimensão |
| `*_run_metadata.json` | modelo, dispositivo, colunas detectadas, quantidade de linhas, tempo e caminhos |

A camada 3 acrescenta `toxicity_score` e `toxicity_label`. A camada 4 acrescenta as
cinco flags multirrótulo.

A comparação produz:

| Arquivo | Conteúdo |
|---|---|
| `cross_layer_verification.csv` | scores lado a lado e três flags de revisão manual |
| `cross_layer_verification_summary.json` | quantidade de casos sinalizados por critério |

As flags de revisão procuram:

1. dimensão da camada 4 ativa em texto considerado não tóxico pela camada 3;
2. toxicidade média/alta sem dimensão discursiva ativa;
3. diferença absoluta de pelo menos 0,35 entre os dois scores de desumanização.

Essas flags não provam erro. Elas organizam a inspeção humana.

## 11. O que pode ser avaliado no arquivo sem gabarito

Com `reddit_tupye_2500_psicologia_sem_gabarito.csv`, é possível verificar:

- se todas as linhas válidas foram processadas;
- se os textos e IDs foram associados corretamente;
- se há scores ausentes ou fora de 0 a 1;
- quantas vezes cada dimensão foi ativada;
- quais comentários têm maior score por dimensão;
- quais hipóteses produziram o score de cada caso;
- quais casos geram divergência entre as camadas;
- tempo total e dispositivo usado.

Não é possível calcular, sem rótulos humanos:

- acurácia;
- precisão;
- recall ou sensibilidade;
- especificidade;
- F1;
- matriz de confusão;
- taxa real de falso positivo ou falso negativo.

Nomear uma frequência de previsão como “acurácia” seria metodologicamente incorreto.

## 12. Inspeção manual recomendada

Depois da execução, carregue as previsões e selecione os extremos e a região do
threshold:

```python
import pandas as pd

df = pd.read_csv("outputs/layer4/layer4_predictions.csv")

top_desumanizacao = df.nlargest(30, "dehumanization_score")
fronteira_desumanizacao = df[
    df["dehumanization_score"].between(0.55, 0.75)
].sample(n=min(50, len(df)), random_state=42)

top_desumanizacao.to_csv(
    "outputs/evaluation/revisao_top_desumanizacao.csv",
    index=False,
    encoding="utf-8-sig",
)
fronteira_desumanizacao.to_csv(
    "outputs/evaluation/revisao_fronteira_desumanizacao.csv",
    index=False,
    encoding="utf-8-sig",
)
```

Repita para as outras dimensões. Revise também amostras aleatórias de scores baixos,
pois elas são necessárias para encontrar falsos negativos.

## 13. Avaliar quando houver gabarito humano

O script `evaluation/evaluate_with_gold.py` calcula métricas binárias para uma
dimensão. Exemplo, supondo que o gabarito tenha `comment_id` e
`gold_dehumanization`:

```bash
python -m evaluation.evaluate_with_gold \
  --predictions outputs/layer4/layer4_predictions.csv \
  --gold data/gabarito_camada4.csv \
  --prediction-id-column source_id \
  --gold-id-column comment_id \
  --predicted-column dehumanization_active \
  --gold-column gold_dehumanization \
  --output outputs/evaluation/dehumanization_metrics.json
```

Faça a avaliação separadamente para cada dimensão, porque a camada 4 é multirrótulo.

## 14. Calibração dos thresholds

Os valores `0.65` da camada 4 e os thresholds da camada 3 são iniciais. O procedimento
adequado é:

1. selecionar uma amostra humana que contenha casos positivos, negativos, casos
   limítrofes e exemplos de citação, negação, humor e discussão acadêmica;
2. obter ao menos dois anotadores por comentário usando as definições do manual;
3. medir concordância e resolver divergências por adjudicação;
4. separar os dados em conjunto de calibração e conjunto de teste antes de ajustar
   qualquer threshold;
5. no conjunto de calibração, testar thresholds por dimensão, por exemplo de 0,05 a
   0,95 em passos de 0,01;
6. escolher a métrica de acordo com o objetivo: recall alto para triagem; precisão
   alta para publicação automática; F1 ou balanced accuracy para compromisso;
7. congelar os thresholds escolhidos;
8. calcular as métricas somente uma vez no conjunto de teste;
9. informar intervalo de confiança por bootstrap, especialmente nas dimensões raras;
10. registrar versão das hipóteses, pesos, modelo e thresholds em cada execução.

Não calibre e teste no mesmo conjunto. Isso produziria uma estimativa otimista.

## 15. Modo offline

Na primeira execução, o Transformers baixa o checkpoint. Depois que ele estiver no
cache local, defina nos dois YAMLs:

```yaml
local_files_only: true
```

Se `local_files_only` estiver ativo antes do download, a execução falhará informando
que os arquivos do modelo não foram encontrados.

## 16. Limitações do checkpoint escolhido

Português não aparece entre as quinze línguas do XNLI listadas no model card desse
checkpoint, embora o XLM-R base tenha pré-treinamento multilíngue mais amplo e possa
generalizar para outros idiomas. Isso torna obrigatória a validação em português
brasileiro e, especialmente, no domínio do Reddit.

Probabilidades NLI não são probabilidades calibradas de verdade social. Scores altos
podem refletir similaridade entre o comentário e a formulação da hipótese. Por isso:

- as hipóteses precisam permanecer versionadas;
- exemplos NEG e NEU precisam ser avaliados;
- thresholds não devem ser escolhidos apenas por intuição;
- ironia, contexto conversacional e referências implícitas continuarão difíceis;
- produção de factualidade não verifica a veracidade da alegação;
- o classificador textual não analisa imagens, memes ou vídeo presentes apenas por
  link.

## 17. Erros comuns

### `CSV não encontrado`

Confirme o nome exato do arquivo ou passe `--input CAMINHO`.

### `Não foi possível detectar a coluna de texto`

Execute com `--text-column NOME`.

### `CUDA out of memory`

Reduza `model.batch_size` nos YAMLs. Feche outros processos que usam a GPU. Se
necessário, defina `device: cpu`.

### Erro ao carregar o tokenizer

Confirme que `sentencepiece` está instalado e reinicie o ambiente Python.

### Saídas já existem

Use `--overwrite` somente quando quiser substituir a execução anterior.

### Execução completa muito lenta

Faça primeiro `--limit 10`, confirme que CUDA está ativa, reduza o número de
processos concorrentes e ajuste o batch ao limite da GPU. Não remova hipóteses apenas
para acelerar sem registrar uma nova versão metodológica.

## 18. Rodar os testes que não dependem do download do modelo

```bash
pytest -q
```

Os testes verificam a agregação, o efeito das contraprovas, a validade dos YAMLs, a
unicidade dos códigos e o total esperado de 40 hipóteses na camada 3 e 63 na camada
4.
