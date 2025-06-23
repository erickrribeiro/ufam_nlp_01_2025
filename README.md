# 🎯 Fine-Tuning em LLMs para Text-to-SQL e Avaliação no MMLU

Este repositório contém um pipeline completo para avaliar o impacto do **fine-tuning via LoRA** em Modelos de Linguagem de Grande Porte (LLMs), aplicado na tarefa de **Text-to-SQL** (Spider) e na avaliação de generalização no benchmark **MMLU (Massive Multitask Language Understanding)**.

---

## 📑 Descrição das Fases do Projeto

### 🔹 **Fase 1: Estabelecimento do Baseline**
- **1.1 Prompt Engineering:**  
Criação de um prompt few-shot contendo 3 exemplos representativos. Este template é fixo para todas as avaliações de baseline.
  
- **1.2 Avaliação do Modelo Base:**  
Submissão do modelo **sem fine-tuning** ao Spider (dev split) utilizando o prompt elaborado.

- **1.3 Coleta de Dados:**  
Armazenamento das queries SQL geradas e cálculo da taxa de sucesso/falha.

---

### 🔸 **Fase 2: Execução do Fine-Tuning**
---

### 🧠 **Fase 3: Avaliação Customizada na Tarefa-Alvo**
- **3.1 Implementação de Métrica Customizada:**  
Desenvolvimento da métrica **Execution Accuracy** no framework **DeepEval**, com as seguintes etapas:
  - Execução da query SQL gerada (`actual_output`) e da ground truth (`expected_output`) em um banco SQLite com os dados do Spider.
  - Comparação dos resultados de forma **order-insensitive**.
  - Retorno de **1.0 (sucesso)** ou **0.0 (falha)**.

- **3.2 Avaliação Automatizada:**  
Integração da métrica em testes via **Pytest**, rodando avaliações completas no Spider dev split.

---

### 📉 **Fase 4: Análise de Regressão de Capacidade (Generalização)**
- **4.1 Avaliação no MMLU:**  
Execução de 150 questões do benchmark MMLU em modo **4-shot**, tanto para o modelo base quanto para os modelos fine-tuned.

- **4.2 Cálculo de Acurácia:**  
Medição da acurácia como proporção de respostas corretas.

- **4.3 Análise de Regressão:**  
Cálculo da **variação percentual de acurácia** entre o modelo base e os modelos fine-tuned:
  - Análise global.
  - Análise por categoria: **STEM**, **Humanidades**, **Ciências Sociais**.

---

## 🚀 Como Executar os Experimentos

### ✅ **Requisitos**
- Conta no **Google Colab**.
- **GPU NVIDIA A100 (40 GB)** alocada no Colab.
- Dependências listadas em `requirements.txt`.

### 🔥 **Passos**
1. Clone ou faça fork deste repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio```

2. No Google Colab:
   * Abra cada notebook na seguinte ordem:
     * `fase-01.ipynb`
     * `fase-02.ipynb`
     * `fase-03.ipynb`
     * `fase-04.ipynb`
   * Verifique se a GPU A100 (40 GB) está ativa:
     **Ambiente de execução → Alterar tipo de hardware → GPU → A100**

3. Execute cada notebook sequencialmente para:

   * Criar o baseline.
   * Realizar o fine-tuning com diferentes hiperparâmetros.
   * Avaliar na tarefa Text-to-SQL com a métrica customizada.
   * Avaliar no benchmark MMLU para análise de regressão.
---

## 📂 Estrutura dos Arquivos

```
├── custom_metrics/         # Métricas customizadas (Execution Accuracy)
├── models/                 # Adaptadores já treinados para não gastar muito com GPU
├── scripts/                # Scripts auxiliares e análise
├── fase-01.ipynb           # Baseline Text-to-SQL
├── fase-02.ipynb           # Fine-tuning com LoRA
├── fase-03.ipynb           # Avaliação com métricas customizadas
├── fase-04.ipynb           # Avaliação no MMLU
├── requirements.txt        # Dependências
└── README.md               # Este arquivo
```
