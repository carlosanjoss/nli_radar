# Dados de entrada

Coloque nesta pasta o arquivo:

`reddit_tupye_2500_psicologia_sem_gabarito.csv`

O leitor tenta detectar automaticamente as colunas de texto `cleaned_body`, `body`,
`text`, `texto`, `comment`, `comentario`, `comentário` ou `content`. Se nenhuma delas
existir, execute com `--text-column NOME_DA_COLUNA`.

O arquivo sem gabarito permite inferência, inspeção dos scores, distribuição dos
rótulos e seleção de casos para revisão. Ele não permite calcular acurácia, precisão,
recall, F1 ou matriz de confusão.

