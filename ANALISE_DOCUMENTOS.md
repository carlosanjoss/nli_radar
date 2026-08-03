# Análise dos documentos e decisões de implementação

## 1. Arquitetura descrita no manual

O arquivo `Manual_ O Que é Discurso de ódio_ Definição e critérios.docx` define
quatro camadas sequenciais e complementares:

1. decisão binária sobre presença ou ausência de discurso de ódio;
2. identificação multirrótulo das categorias de preconceito;
3. estimação da intensidade ou toxicidade;
4. identificação dos mecanismos retóricos, enquadramentos ideológicos e
   estratégias discursivas.

O manual estabelece um gate conceitual: conteúdo não classificado como discurso
de ódio na camada 1 deveria encerrar a análise. Portanto, quando a camada 4 é
executada sobre toda a amostra sem uma coluna de gate, o resultado deve ser tratado
como exploração e triagem, não como classificação final do Radar.

## 2. Diferenças entre os dois documentos da camada 4

O arquivo `Proposta Camada4 Dimensoes_Discursivas.docx` não está completo o
suficiente para implementação integral:

- o título anuncia cinco dimensões, mas a tabela apresenta somente quatro;
- a linha de conspiração e desinformação não possui núcleo semântico nem referência;
- somente desumanização possui hipóteses NLI desenvolvidas;
- as tabelas de autoritarismo estão vazias;
- revitimização não possui hipóteses no documento curto;
- a quinta dimensão anunciada não é identificada.

O manual completo apresenta cinco dimensões com definições, construtos e hipóteses:

| Código implementado | Dimensão do manual | Construtos |
|---|---|---|
| `dehumanization` | Desumanização | animalizante; mecanicista; desengajamento moral |
| `authoritarianism` | Autoritarismo | submissão à autoridade; agressividade punitiva; convencionalismo excludente |
| `revictimization` | Revitimização e autovitimização | culpabilização; negação/minimização; inversão vítima-agressor |
| `factualization` | Produção de factualidade | factualização dubitativa; factualização categórica |
| `masculinism` | Masculinismo | ressentimento; emasculação; antifeminismo; supremacia masculina |

Por isso, `layer4/hypotheses.yaml` foi construído com base no manual completo. O
documento curto foi utilizado como confirmação da proposta inicial, mas não como
fonte exclusiva.

## 3. Produção de factualidade não é verificação de desinformação

A proposta curta menciona “conspiração e desinformação”, mas não define o
construto. O manual completo desenvolve “produção de factualidade”. Os conceitos
não são equivalentes.

A dimensão implementada identifica como uma acusação contra um grupo é apresentada
como suspeita, evidência, estatística, fato incontestável, pergunta pressuposta,
relato autorizado ou prova multimodal. Ela não decide se a alegação é verdadeira
ou falsa. `xlm-roberta-large-xnli` compara semanticamente uma premissa com uma
hipótese; ele não consulta fontes e não executa fact-checking.

Consequentemente, o código usa o nome `factualization` e não
`misinformation`. Uma futura dimensão de desinformação precisará de definição
operacional própria e, para verificar veracidade, de recuperação de evidências,
fontes confiáveis e outro estágio de decisão.

## 4. Inconsistência encontrada no código informado como camada 3

O código fornecido pelo usuário é chamado de camada 3, mas contém:

- `CONFIG_PATH = "config/layer4_config.yaml"`;
- tabelas `reddit_nli_layer4_runs`, `reddit_nli_layer4_hypotheses`,
  `reddit_nli_layer4_scores` e `reddit_nli_layer4_final_scores`;
- nome de execução `layer4_sample_*`.

Esses nomes misturam as camadas. No novo pacote, os arquivos e resultados da lógica
existente foram colocados exclusivamente em `layer3/` e `outputs/layer3/`.

Há ainda uma diferença conceitual. O manual define cinco níveis de hostilidade:
preconceito implícito, hostilidade explícita, desumanização, incitação à violência
e extermínio. O código existente produz quatro faixas (`nao_toxico`, `baixa`,
`media`, `alta`) a partir de oito sinais (`disrespect`, `insult`, `humiliation`,
`dehumanization`, `violence`, `exclusion`, `identity_attack`, `threat`).

Esta entrega preserva a lógica operacional do código existente para não alterar uma
camada que o usuário declarou pronta. A adequação integral da camada 3 aos cinco
níveis do manual deve ser tratada como uma etapa posterior, com validação humana e
recalibração de thresholds.

## 5. Correção do cálculo de probabilidade NLI

O código anterior aplica `softmax` simultaneamente a entailment, neutral e
contradiction e usa diretamente a probabilidade de entailment. O model card do
checkpoint escolhido mostra, para classificação zero-shot multirrótulo, o descarte
do logit neutral e a normalização entre contradiction e entailment:

\[
P(\text{hipótese verdadeira}) =
\frac{e^{z_E}}{e^{z_E}+e^{z_C}}.
\]

O pacote grava as duas visões:

- as três probabilidades do softmax completo, para auditoria;
- `label_probability`, calculada por entailment versus contradiction e usada na
  agregação.

Para reproduzir o comportamento antigo, é possível definir
`score_mode: three_way_entailment`, mas esse modo não é o padrão.

## 6. Hipóteses positivas e contraprovas

O manual fornece, além das hipóteses positivas, hipóteses `NEG` e `NEU`. Ignorá-las
perderia as fronteiras conceituais usadas para diferenciar ataque de crítica,
endosso de citação, acusação de descrição e hostilidade de análise acadêmica.

O pacote calcula, para cada construto:

1. score positivo bruto: 70% do maior score positivo e 30% da média positiva;
2. contraprova: maior score entre as hipóteses negativas e neutras;
3. score ajustado: score positivo bruto menos 50% da contraprova, limitado ao
   intervalo de 0 a 1.

Os pesos são explícitos em cada `config.yaml`. Eles são escolhas iniciais para
teste, não parâmetros validados empiricamente.

## 7. Natureza multirrótulo da camada 4

As dimensões são transversais. Um comentário pode ativar simultaneamente, por
exemplo, desumanização, autoritarismo e produção de factualidade. Por isso, o
classificador não aplica `argmax` como decisão exclusiva. Ele registra:

- score contínuo de cada dimensão;
- indicador booleano por dimensão;
- lista de todas as dimensões ativas;
- dimensão primária apenas como conveniência para inspeção.

## 8. Limites da avaliação sem gabarito

O arquivo solicitado contém `sem_gabarito` no nome. Sem rótulos humanos, não é
possível afirmar que o classificador possui determinada acurácia, precisão, recall
ou F1. A verificação incluída é descritiva e produz:

- distribuição dos scores;
- frequência de ativação por dimensão;
- dimensão predominante;
- comparação entre as camadas 3 e 4;
- casos prioritários para revisão manual.

Os casos sinalizados na verificação cruzada não são automaticamente erros. Eles são
observações em que as duas camadas produziram padrões que merecem inspeção.

