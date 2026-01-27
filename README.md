# Plataforma de Machine Learning para Precificação Dinâmica (Dataset Kaggle Olist)

## Visão Geral
Este projeto implementa uma **plataforma de Machine Learning pronta para produção**, utilizando **dados reais de e-commerce** do Kaggle (dataset Olist), com foco em:

- Precificação dinâmica
- Engenharia de atributos reutilizável
- Pipeline de treinamento em batch
- API de inferência em tempo real
- Arquitetura preparada para escalar (Spark, Kafka, MLflow)

O objetivo é simular **um sistema real utilizado por empresas de tecnologia**, indo além de projetos acadêmicos ou tutoriais.

---

## Dataset Utilizado (Kaggle)

**Brazilian E-Commerce Public Dataset by Olist**

- Fonte: Kaggle
- Volume: ~100 mil pedidos
- Domínio: Marketplace / E-commerce real

### Arquivo utilizado
- `olist_order_items_dataset.csv`

### Principais colunas
- `price` — preço do item
- `freight_value` — custo de frete
- `order_id`
- `product_id`
- `seller_id`

O dataset é usado para **simular um cenário real de precificação dinâmica**, criando uma variável target baseada em demanda e custo logístico.

### Baixar dataset do Kaggle via Powershell
```
python -m pip install kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce
Expand-Archive -Path brazilian-ecommerce.zip -DestinationPath .
Move-Item olist_order_items_dataset.csv data\
```

---

## Problema de Negócio
Marketplaces precisam ajustar preços considerando:
- Custos logísticos
- Demanda implícita
- Margem de lucro
- Estratégias comerciais

Neste projeto, treinamos um modelo para **estimar um preço dinâmico ajustado**, simulando decisões de pricing em produção.

---

## Arquitetura do Sistema

```
[ Dataset Kaggle (Olist) ]  
            ↓  
[ Feature Store (Python) ]  
            ↓  
[ Pipeline de Treinamento Batch ]  
            ↓  
[ Modelo de Machine Learning ]  
            ↓  
[ API REST (FastAPI) ]  
            ↓  
[ Predição em Tempo Real ]
```


---

## Stack Tecnológica

- **Python**
- **SQL (conceitual, extensível)**
- pandas, numpy
- scikit-learn
- XGBoost
- FastAPI
- Docker (opcional)
- Cloud-ready (AWS / GCP / Azure)

---

## Estrutura do Repositório

```text
dynamic-pricing-ml-platform/
│
├── api/
│   └── main.py
│
├── feature_store/
│   └── build_features.py
│
├── models/
│   └── gbm_model.py
│
├── pipelines/
│   └── batch_training.py
│
├── data/
│   └── olist_order_items_dataset.csv
│
├── requirements.txt
├── Dockerfile
└── README.md
```


---

## Feature Engineering
As features são criadas de forma **padronizada**, garantindo consistência entre treino e produção:

- `price`
- `freight_value`
- `total_value` (price + freight)
- `high_freight` (proxy de custo/demanda)

---

## Modelo
- **Gradient Boosting (XGBoost)**
- Robusto, interpretável e amplamente usado em produção
- Preparado para evoluir para ensemble ou deep learning

---

## Como Executar o Projeto

### 1 Instalar dependências
- Criar o ambiente virtual
```
python -m venv venv
```
- Ativr o ambiente virtual no Powershell
```
venv\Scripts\activate
```
- Criar arquivo requirements.txt
```
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.0
fastapi==0.110.0
uvicorn==0.27.1
joblib==1.3.2
```
- Instalar as dependências
```
pip install -r requirements.txt
```
- Validar a Instalação
```
python -c "import pandas, numpy, fastapi, scikit-learn, fastapi, uvicorn, joblib"
```

### 2 Treinar o modelo
```
python pipelines/batch_training.py
```

### 3 Subir a API
```
uvicorn api.main:app --reload
```
- Abra:
```
http://127.0.0.1:8000/docs
```

### 4 Fazer uma predição
```
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"price": 120.0, "freight_value": 18.5}'
```

## Deploy com Docker
- Criar o Dockerfile
```
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- Build da imagem Docker
```
docker build -t dynamic-pricing-ml .
```
- Rodar container
```
docker run -p 8000:8000 dynamic-pricing-ml
```
- API disponível
```
http://localhost:8000/docs
```

## Projeto para Currículo

Desenvolvimento de uma plataforma de Machine Learning para precificação dinâmica utilizando dados reais de e-commerce, com pipeline de treinamento em batch, feature engineering reutilizável e API de inferência em produção.