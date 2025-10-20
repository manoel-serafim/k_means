# Implementação K-Means

Este repositório contém uma implementação paralela do algoritmo de clusterização K-Means com versões sequencial e paralela usando OpenMP, projetada para análise de desempenho em diferentes configurações de threads.

## Requisitos

O projeto requer distribuição Linux baseada em Debian com Clang e suporte OpenMP. Instale os pacotes: `clang`, `libomp-dev` e `make`.

## Compilação

Para compilar o projeto execute:
```bash
clear && make clean && make
```

Os executáveis são gerados em `build/bin`: versão sequencial para baseline e versão OpenMP com suporte a múltiplas threads.

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