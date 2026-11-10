# Nota: letras reconhecidas pela forma da mão (`hand_shape_v1`)

Data: 2026-11-10 · Substitui o diagnóstico de `NOTA_UMA_MAO_MODELO_HOLISTICO.md`.

## Sintoma
O app trocava A e B por I, D e C, mesmo com o modelo `HOLÍSTICO_LIBRAS` marcando 99% de acurácia.

## Causa
O modelo estático era treinado com o vetor holístico inteiro (269 valores: posição do pulso na tela,
mão, corpo e rosto). Para uma letra só a forma da mão importa; o resto descreve quem gravou e onde
estava. A rede decorou isso.

Os 99% vinham de um teste sorteado entre quadros vizinhos da mesma gravação, que são quase iguais.

Medido com as amostras de 08/11 (outra pessoa, gravadas depois do treino):

| Letra | Modelo antigo | Forma da mão |
|---|---|---|
| A | 78% | 100% |
| B | 0% | 100% |
| C | 33% | 100% |
| I | 12% | 100% |
| N | 100% | 100% |

## O que mudou
- **Entrada do modelo:** 165 valores de uma mão — pontos relativos ao pulso, divididos pelo tamanho
  da mão, mais distâncias entre pontas e nós dos dedos (`src/data_collection/hand_shape_features.py`).
  Não depende de posição na tela, distância da câmera, corpo ou rosto.
- **Treino** (`src/training/static_trainer.py`): datasets estáticos holísticos são convertidos sozinhos
  (a mão crua já estava gravada). O treino gira, espelha e adiciona ruído às amostras, e equilibra as
  letras com poucas amostras.
- **Acurácia:** medida em sessões de coleta que o treino não viu. O relatório avisa quando uma letra
  tem menos de 2 sessões ou de 60 amostras.
- **Inferência:** cada mão do quadro é classificada sozinha. Funciona com o modo holístico do app
  ligado ou desligado.
- **Mão duplicada:** o MediaPipe às vezes devolve a mesma mão duas vezes. `parse_input` descarta a
  cópia (`dedupe_hands`).

Datasets "só mãos" (42 valores) e modelos já publicados continuam funcionando como antes.

## Como publicar
1. Subir esta versão da API (o servidor antigo não sabe ler o modelo novo).
2. Treinar de novo: `treinar.bat`, ou
   `python scripts/train_local.py --model HOLÍSTICO_LIBRAS --datasets ALFABETO_HOLÍSTICO`.
3. Ativar o modelo no app.

## Limite
O modelo só conhece as letras do dataset (hoje A, B, C, D, E, F, I, N; F tem 31 amostras de uma
sessão). Uma letra fora dele é respondida como a mais parecida entre as conhecidas.
