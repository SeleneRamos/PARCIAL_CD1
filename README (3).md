# Aplicación de Redes Neuronales para la clasificación de ingresos anual (> $50K)

Este repositorio contiene un proyecto enfocado en el ciclo completo de diseño, optimización, interpretación y despliegue de un modelo de Machine Learning basado en una red neuronal profunda tipo **Perceptrón Multicapa (MLP)** para datos tabulares estructurados.

## 👥 Integrantes del Equipo
* **David Barrientos**
* **Marcello Anchante**
* **Sebastian Rios**
* **Selene Ramos**

---

## 🎯 Objetivo del Proyecto
Construir, ajustar, interpretar y desplegar un modelo predictivo basado en redes neuronales para clasificar si los ingresos anuales de un individuo superan los \$50,000 dólares (`>50K` o `<=50K`), utilizando el conjunto de datos histórico **Adult / Census Income Dataset** de la Oficina de Censos de los EE. UU. (1994).

El proyecto demuestra el dominio de:
`Preprocesamiento` ➡️ `Ingeniería de Variables` ➡️ `Modelado & Tuning` ➡️ `Ablation Study` ➡️ `Evaluación & Interpretabilidad` ➡️ `Despliegue Interactivo`

---

## 📂 Estructura del Repositorio

El proyecto está organizado de manera secuencial siguiendo los requerimientos estrictos de la rúbrica del examen:

```text
├── 01_eda.ipynb                           # Análisis Exploratorio de Datos y Calidad de Datos
├── 02_preprocesamiento_feature_engineering.ipynb # Limpieza, codificación y creación de variables (No Leakage)
├── 03_modelo_keras.py                     # Implementación y tuning de la red neuronal en Keras
├── 04_modelo_pytorch_o_sklearn.py         # Implementación alternativa en PyTorch / Scikit-Learn
├── 05_ablation_study.ipynb                # Estudio de ablación comparativo incremental
├── 06_evaluacion_interpretabilidad.ipynb  # Métricas avanzadas, análisis de errores y SHAP/Permutations
├── app_streamlit/                         # Carpeta de la aplicación interactiva para predicción en vivo
│   └── app.py                             # Código principal de la interfaz en Streamlit
├── modelo_final/                          # Modelos entrenados y serializados (pesos y pipelines)
├── requirements.txt                       # Dependencias del proyecto (pip)
└── README.md                              # Documentación general del proyecto (este archivo)
```

---

## 🛠️ Requisitos e Instalación

Para asegurar la **reproducibilidad** total del proyecto (punto crítico de penalización), siga los siguientes pasos para configurar su entorno local:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Marcelloprime/rnn_adult_income_prediction.git
   cd rnn_adult_income_prediction
   ```

2. **Crear y activar un entorno virtual (Recomendado):**
   * En Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * En Windows (Git Bash / PowerShell):
     ```bash
     python -m venv venv
     source venv/Scripts/activate
     ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
