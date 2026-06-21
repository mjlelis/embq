# Arquitetura e Decisões Técnicas

## Visão Geral do Sistema
O `embq` é dividido em três camadas principais para equilibrar performance extrema com facilidade de uso:

### 1. Kernel Rust (`embq-core`)
Localizado em `crates/embq-core`, este é um pacote Rust puro, sem dependências de runtime Python.
- **Estruturas de Dados:** Utiliza layouts de memória contíguos (Row-major) para garantir localidade de cache.
- **Normalização L2:** Implementada para que a similaridade de cosseno possa ser calculada via produto escalar simples, reduzindo ciclos de CPU.
- **Autovetorização:** O código foi escrito de forma idiomática para que o compilador Rust (LLVM) possa aplicar otimizações SIMD (AVX2, AVX-512) automaticamente, sem a necessidade de código intrínseco de difícil manutenção.

### 2. Binding PyO3 (`embq-py`)
Localizado em `crates/embq-py`, atua como uma ponte fina.
- **Zero Cópia (sempre que possível):** Utiliza as facilidades do `numpy-rust` para acessar os buffers de memória do NumPy diretamente, minimizando o overhead de transferência de dados entre Rust e Python.
- **Abstração de Erros:** Converte erros específicos do Rust (`thiserror`) em exceções Python amigáveis.

### 3. Casca Python (`python/embq`)
A camada de interface com o usuário.
- **I/O:** Suporte nativo para `.npy` (NumPy) e `.parquet` (via PyArrow/Pandas), que são os padrões da indústria para datasets de embeddings.
- **Relatórios:** Utiliza a biblioteca `tabulate` para gerar tabelas comparativas legíveis no terminal.

## Por que Rust?
A escolha por Rust não é apenas por velocidade bruta.
- **Segurança de Memória:** Garantia de que operações de baixo nível com bits (na quantização binária) não causem crashes ou vazamentos de memória.
- **Gerenciamento de Recursos:** O Rust permite um controle granular sobre como os dados são empacotados (ex: `u64` para bits de quantização binária), o que é essencial para medir compressão real.

## Por que Python?
Python é a língua franca da Inteligência Artificial.
- **Ecossistema:** Facilidade de carregar dados de diversas fontes e integrar com ferramentas de visualização.
- **Iteração Rápida:** Permite que o usuário experimente diferentes parâmetros de `oversample` ou `k` sem precisar recompilar o projeto.
