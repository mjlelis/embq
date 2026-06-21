# Guia Técnico de Quantização

Este guia explica os métodos implementados no `embq` e como eles afetam seus resultados.

## 1. FP32 (Full Precision)
- **O que é:** O formato original dos embeddings gerados por modelos como BERT, CLIP ou Ada-002.
- **Armazenamento:** 4 bytes por dimensão.
- **Uso no embq:** Serve como o "Oráculo". Todos os outros métodos são comparados contra ele para calcular o Recall.

## 2. INT8 (Quantização Escalar)
- **O que é:** Converte os valores contínuos de ponto flutuante para inteiros de 8 bits (-128 a 127).
- **Como funciona:** O `embq` calcula uma escala global simétrica baseada no valor máximo absoluto (`amax`) dos embeddings e mapeia os valores para o intervalo de 8 bits.
- **Vantagem:** Redução de 4x no consumo de memória com perda de precisão quase imperceptível (geralmente Recall > 0.99).

## 3. Binary (1-bit Quantization)
- **O que é:** Cada dimensão do vetor é reduzida a um único bit.
- **Como funciona:** Se o valor for maior que zero, o bit é 1; caso contrário, é 0.
- **Armazenamento:** 1 bit por dimensão (redução de 32x em relação ao FP32).
- **Busca:** Utiliza a **Distância de Hamming**, que é extremamente rápida em CPUs modernas usando instruções de `XOR` e `popcount`.
- **Limitação:** Pode haver uma perda significativa de precisão (o "penhasco do binário"), especialmente para modelos que não foram treinados especificamente para quantização binária.

## 4. Binary + Rescore (Re-ranking)
- **O que é:** Uma técnica de duas fases para recuperar a precisão da quantização binária.
- **Fase 1 (Candidatos):** Realiza uma busca ultra-rápida no índice binário para encontrar os top `k * oversample` vizinhos (ex: se k=10 e oversample=20, busca 200 candidatos).
- **Fase 2 (Re-ranking):** Para esses candidatos, acessa os vetores FP32 originais e recalcula a similaridade exata.
- **Vantagem:** Mantém a velocidade da busca binária na maior parte do processo, mas entrega um Recall muito próximo do original, com uma pegada de memória muito menor para o índice principal.

## Métricas Medidas
- **Recall@K:** A porcentagem dos "vizinhos verdadeiros" (do FP32) que aparecem na lista de resultados do método quantizado.
- **Latência:** Tempo médio gasto por consulta (query).
- **Compressão:** Proporção entre o tamanho original (FP32) e o tamanho quantizado.
- **Bytes/vec:** Quantidade real de memória necessária para armazenar um único vetor no formato especificado.
