# Implementação K-Means

[!] As implementações em CUDA e MPI podem ser facilmente reexecutadas através dos .ipynb disponíveis nesse repositório.

Este repositório contém uma implementação paralela do algoritmo de clusterização K-Means com versões sequencial e paralela usando OpenMP e CUDA projetada para análise de desempenho em diferentes configurações. A implementação dos testes levou em conta que a magnitude dos pontos também deveria ter variância significativa para que o número de iterações computadas fosse grande o suficiente. 

## Requisitos

O projeto requer distribuição Linux baseada em Debian com Clang e suporte OpenMP. Instale os pacotes: `clang`, `libomp-dev` e `make`.

## Compilação

Para compilar o projeto execute:
```bash
clear && make clean && make
```

Os executáveis são gerados em `build/bin`: versão sequencial para baseline e versão OpenMP.

## Testes e Avaliação
Nessa mesma pasta
Execute a suite de testes completa com múltiplos datasets e configurações de threads:
```bash
python test/test.py
```

Após os testes, gere relatórios de desempenho com métricas de speedup, eficiência e validação SSE:
```bash
python test/evaluate.py
```

## Resultados

Os resultados são salvos em `test/measurements` incluindo métricas de desempenho, logs de execução e gráficos de visualização. A estrutura separa medições sequenciais e OpenMP, com cada configuração mantendo arquivos de atribuições de clusters, centroides finais e estatísticas de execução.