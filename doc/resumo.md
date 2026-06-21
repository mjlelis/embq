# Resumo do Projeto: embq

## O que é?
O `embq` é uma ferramenta de perfilamento (profiling) de trade-offs de quantização para embeddings. Ele foi projetado para ajudar engenheiros de IA e busca vetorial a entenderem o impacto de diferentes técnicas de compressão (como INT8 e Quantização Binária) na precisão (Recall), latência e consumo de memória de seus modelos, utilizando seus próprios dados reais.

## Por que existe?
No mundo real, armazenar e pesquisar bilhões de vetores de alta dimensionalidade (embeddings) em `float32` é extremamente caro e lento. A quantização é a solução padrão, mas cada modelo de embedding reage de forma diferente à compressão. Alguns mantêm 99% da precisão com INT8, outros sofrem um "penhasco de precisão" com quantização binária.

O `embq` permite que você tome decisões baseadas em dados em vez de suposições, respondendo perguntas como:
- "Quanto de memória eu economizo se mudar para INT8?"
- "Qual é a perda de Recall@10 se eu usar quantização binária com re-ranking (rescore)?"
- "Qual o ganho de velocidade real na minha CPU?"

## Como funciona?
O projeto utiliza uma arquitetura híbrida:
1.  **Core em Rust (`embq-core`):** Todo o trabalho pesado de cálculo matemático, manipulação de bits (Hamming distance) e algoritmos de busca é feito em Rust. Isso garante performance de nível de sistema e aproveita as instruções SIMD modernas da CPU de forma automática.
2.  **Interface Python (`embq-py`):** Através do PyO3, o motor Rust é exposto como uma biblioteca Python nativa. Isso permite que cientistas de dados usem o `embq` diretamente em seus notebooks ou scripts de avaliação com objetos NumPy e arquivos Parquet.
3.  **CLI Intuitiva:** Uma interface de linha de comando em Python permite rodar testes rápidos em arquivos de dados sem escrever uma única linha de código.

## Principais Métodos Avaliados
- **FP32:** A linha de base (Ground Truth) sem compressão.
- **INT8:** Reduz o tamanho do vetor em 4x com perda mínima de precisão na maioria dos casos.
- **Binary (1-bit):** Reduz o tamanho em até 32x, usando apenas o sinal do vetor. Extremamente rápido via operações de XOR e popcount.
- **Binary + Rescore:** O melhor dos dois mundos. Usa a busca binária ultra-rápida para filtrar candidatos e depois reordena o topo da lista usando os vetores originais (FP32), recuperando grande parte da precisão perdida.
