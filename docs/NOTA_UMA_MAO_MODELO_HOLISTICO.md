# Nota: modelo estático holístico só funciona com uma mão

Data: 2026-11-08 · Modelo: `HOLÍSTICO_LIBRAS` (estático) · Dataset: `ALFABETO_HOLÍSTICO`

## Sintoma
O app reconhece **C** quase o tempo todo e tem dificuldade para trocar para outras letras.

## Causa (diagnosticada, ainda não corrigida)
O dataset de treino tem **uma mão só**: 0% das amostras de A, B, C, D, I e N têm a mão 2
(só 1% das de E). O vetor holístico reserva um bloco para a mão 2, que fica zerado quando
ela não aparece.

Na inferência, se o MediaPipe detecta duas mãos (a outra mão, ou uma detecção falsa), o bloco
da mão 2 vem preenchido. É um padrão que o modelo nunca viu, e ele responde **C com confiança alta**
(~0,95), então o `min_score` não filtra.

## Evidência
- Teste no modelo baixado, com 60 amostras reais por letra:
  - 1 mão: acerto de 95% a 100% em todas as letras.
  - Com um segundo bloco de mão injetado: acerto de 0% a 17% (exceto C, 72%), e **65% das previsões viram C**.
  - Com a segunda mão copiada da primeira: **67% viram C**.
- Log de inferência anterior com esse modelo: 489 respostas com 2 mãos, e o servidor exibiu C em exatamente 489.
  Não foi possível cruzar frame a frame (o arquivo foi substituído), então é indício.
- Limite do teste: a segunda mão foi simulada, e as amostras de teste vêm do próprio treino.

## Decisão
**Não alterar código por enquanto.** Testar usando **apenas uma mão** na câmera e ver se as trocas de letra
funcionam. Se funcionarem, a hipótese está confirmada.

## Como validar
1. Ligar o log de inferência (Configurações) e usar uma mão só, trocando A, B, D, A, B, D (~3 s cada).
2. Conferir no log que `hands` é `1` nas respostas e que o gesto exibido acompanha a letra.
3. Se aparecer `hands: 2` com uma mão só, há segunda detecção falsa do MediaPipe.

## Se for preciso corrigir depois
- **Servidor (recomendado):** no caminho do modelo estático holístico, avaliar cada mão separadamente
  como mão 1 (bloco 2 zerado) e ficar com a de maior confiança. Não exige novo APK.
- **App:** `numHands: 1` no plugin do MediaPipe em `app.json`. Exige refazer o build nativo.
- **Retreino com amostras de duas mãos:** só se duas mãos forem um caso válido do produto.

## Observação
O cooldown de 1 s do `DetectorManager` (+ 3 frames de estabilidade) também atrasa a troca de letra,
mas é um fator secundário e não explica o domínio do C.
