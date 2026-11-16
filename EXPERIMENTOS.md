# Experimentos não adotados

Cada estudo fica em `analysis/` e grava em `reports/analysis/`. Nenhum usa o teste. A regra foi fixada antes de rodar: uma variante só substitui a v1 se ganhar dela em nDCG@10 com IC 95% acima de zero.

## Treinar só com candidatos julgados (não adotado)

O ranker trata produto não julgado como irrelevante. A alternativa é treinar só com os candidatos que têm rótulo (`analysis/02_judged_only.py`, validação). O nDCG condensado não muda (0,836 contra 0,832, IC passando pelo zero), e o nDCG@10 cai para 0,730 (−0,056, IC [−0,085; −0,030]). Mantive o treino com todos os candidatos.

## Tentativa v2 (não adotada)

Testei quatro grupos de features novas na validação:

- classe prevista da busca pelo consenso dos 20 primeiros candidatos (`analysis/03_query_class.ipynb`);
- números com tolerância ("24 inches" casa com 24,75'');
- cor, material e estilo pedidos contra a chave certa do atributo;
- scores normalizados dentro da busca.

Com todas, o nDCG@10 foi 0,792, **+0,006 sobre a v1, IC [−0,006; +0,018]**. O melhor subconjunto (sem os atributos novos) chegou a 0,794, +0,008, IC [−0,003; +0,020]. Nenhum passou na regra de seleção, e o custo não compensa: o p95 do ranking sobe ~1,6× (de ~2,0 para ~3,1 ms na mesma medição), com 136 árvores contra 74. Mantive a v1, e o teste não foi reavaliado. As variantes estão em `analysis/04_v2_features.py` e `reports/analysis/04_v2_features.json`. Só mudar a ordem das colunas (o `colsample_bytree` sorteia por posição) leva a v2 de 0,792 para 0,782: a diferença para a v1 está dentro do ruído do próprio treino.

Também medi as ideias de número e de atributo **só nas buscas que pedem isso**, com validação cruzada em 5 partes sobre treino + validação (`analysis/05_affected_queries.py`). Nas 35 buscas com número e nas 97 que pedem cor, material ou estilo, nenhuma variante ganha da v1 (diferenças entre −0,011 e +0,003, todas com IC passando pelo zero). A v1 já tem `attr_overlap`, que compara as palavras da busca com todos os valores de atributo, então a explicação mais provável é que a informação nova seja quase toda redundante.

## Correções das features da v1 (não adotadas)

Revisei as features da v1 e testei cinco correções com validação cruzada em 5 partes sobre treino + validação (383 buscas, `analysis/06_v3_features.py`). A regra foi fixada antes de rodar: só entra o que ganhar da v1 com IC acima de zero.

| Correção | Problema na v1 | nDCG@10 contra a v1 |
|---|---|---|
| `query_in_name` por palavra inteira | casa pedaço de palavra: "art" bate em "party" | −0,002, IC [−0,005; +0,002] |
| overlaps pesados por IDF | "with" vale o mesmo que "velvet" | **−0,009, IC [−0,016; −0,003]** |
| nota média bayesiana | 62% dos produtos com 1–2 avaliações têm nota 5,0 | 0,000, IC [−0,004; +0,004] |
| atributos só descritivos | garantia e país de origem entram no overlap | 0,000, IC [−0,004; +0,004] |
| fração de palavras fora do catálogo | erros de digitação e marcas cegam o BM25 | −0,001, IC [−0,004; +0,003] |
| todas juntas | — | −0,010, IC [−0,017; −0,004] |

- Nenhuma passou. O bug do `query_in_name` é real, mas corrigir não muda o resultado, provavelmente porque o `name_overlap` (que já compara palavras inteiras) cobre o mesmo sinal. Fica como limitação conhecida da v1.
- O IDF **piora**. Ele dá peso baixo às palavras de tipo de produto, que são comuns no catálogo (`chair` 1,9, `table` 1,7), e peso alto aos modificadores (`velvet` 3,8, `gold` 3,8). Mas o tipo de produto é o que mais separa Exact de Irrelevant (ver o EDA). O IDF inverte essa prioridade.

## Outras alternativas no ranking (validação)

Com as features da v1 (`analysis/07_ranking_alternatives.py`):

| Alternativa | nDCG@10 | contra o final |
|---|---|---|
| **LambdaMART, Exact com ganho 3 no treino (final)** | **0,786** | — |
| LambdaMART, Exact com ganho 2 | 0,792 | +0,006, IC [−0,005; +0,016] |
| LambdaMART, Exact com ganho 10 | 0,777 | −0,009, IC [−0,017; −0,002] |
| XE-NDCG (listwise) | 0,784 | −0,002, IC [−0,013; +0,009] |
| regressão da nota (pointwise) | 0,777 | −0,009, IC [−0,027; +0,011] |
| classificação "é Exact?" (pointwise) | 0,753 | −0,034, IC [−0,060; −0,008] |

- **Peso do Exact:** o LightGBM já usa ganho 3 para o Exact no treino (contra 1 do Partial). Aumentar piora: com 10, o modelo deixa de separar Partial de Irrelevant, e a métrica cobra essa separação. O ganho da métrica (Exact 2, Partial 1) não foi mexido: é definição do problema, não hiperparâmetro.
- **Objetivo:** os dois objetivos de ranking empatam. Os pointwise ficam atrás, porque julgam cada produto sozinho, sem compará-lo com os outros da mesma busca. A classificação fica abaixo até da ordem do retrieval, porque trata Partial igual a Irrelevant.

## Modelo de embedding mais forte (não adotado)

Troquei o `all-MiniLM-L6-v2` pelo `bge-small-en-v1.5` (mesma dimensão, 384), com o prefixo de busca do bge, em duas versões: o texto curto atual (nome, classe e categoria) e um texto longo (+ descrição e atributos). O ranker usa as mesmas 17 features. Retrieval na validação; LTR com CV em 5 partes × 3 seeds sobre treino + validação, com early stopping em 20% das buscas de treino de cada fold (`analysis/09_embedding_model.py`).

| Versão | Denso, Recall@100 | Híbrido, Recall@100 | Híbrido, nDCG@10 | LTR, nDCG@10 (CV) | LTR contra o atual |
|---|---|---|---|---|---|
| MiniLM, texto curto (atual) | 0,589 | 0,677 | 0,758 | 0,798 | — |
| bge-small, texto curto | 0,633 | 0,679 | 0,761 | 0,794 | −0,004, IC [−0,013; +0,005] |
| bge-small, texto longo | 0,660 | 0,691 | 0,756 | 0,805 | +0,007, IC [−0,001; +0,016] |

- **O denso melhora muito, o híbrido quase nada.** Com texto longo, o denso sobe +0,071 de recall (IC [+0,024; +0,124]) e fica perto do BM25 (0,669). Mas no híbrido o ganho cai para +0,014, com IC passando pelo zero: o texto longo traz as mesmas palavras que o BM25 já indexa, então o denso passa a achar o que o BM25 já achava.
- **No LTR, o ganho não passa na regra.** O nDCG@10 sobe +0,007 com IC encostando em zero. O condensado sobe +0,010, IC [+0,002; +0,018], então há sinal, mas pequeno. Só o modelo, com texto curto, piora; o que ajuda é o texto longo. Não testei o MiniLM com texto longo, então não separei o efeito do modelo do efeito do texto.
- **Custo:** a indexação vai de 170 s para ~30 min (~10×), e o encode da busca de ~6 para ~12 ms (p50). Mantive o MiniLM, e o teste não foi reavaliado.
