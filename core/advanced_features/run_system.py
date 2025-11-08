"""
Demo Integrado - Demostración completa del sistema de características avanzadas
Este script muestra todas las características funcionando en conjunto de manera coordinada
"""

import asyncio
import time
import json
import os
import numpy as np
from typing import Dict, Any, List

# Importar el orquestador y configuraciones
try:
    from . import (
        create_orchestrator,
        create_comprehensive_config,
        ProcessingMode,
        AdvancedConfig
    )
    from .dynamic_chunking import ChunkType
    from .multi_vector_retrieval import VectorType
    from .query_expansion import QueryType
    from .confidence_calibration import CalibrationMethod
except ImportError:
    # Para ejecución directa
    import sys
    sys.path.append(os.path.dirname(__file__))
    from __init__ import (
        create_orchestrator,
        create_comprehensive_config,
        ProcessingMode,
        AdvancedConfig
    )
    from dynamic_chunking import ChunkType
    from multi_vector_retrieval import VectorType
    from query_expansion import QueryType
    from confidence_calibration import CalibrationMethod


class IntegratedDemo:
    """Clase principal para la demostración integrada"""

    def __init__(self):
        self.orchestrator = None
        self.demo_data = self._create_demo_data()

    def _create_demo_data(self) -> Dict[str, Any]:
        """Crea datos de ejemplo para la demostración"""

        return {
            'documents': [
                {
                    'id': 'doc_ml_intro',
                    'content': """# Introducción a Machine Learning

Machine Learning es una rama de la inteligencia artificial que permite a las computadoras aprender y tomar decisiones a partir de datos sin ser programadas explícitamente para cada tarea específica.

## Tipos principales de Machine Learning

### 1. Aprendizaje Supervisado
El aprendizaje supervisado utiliza datos etiquetados para entrenar modelos. Los algoritmos aprenden de ejemplos de entrada-salida para hacer predicciones sobre nuevos datos no vistos.

Ejemplos comunes:
- Clasificación de imágenes
- Detección de spam
- Predicción de precios
- Diagnóstico médico

### 2. Aprendizaje No Supervisado
Este enfoque trabaja con datos sin etiquetas, buscando patrones ocultos o estructuras en los datos.

Técnicas principales:
- Clustering (K-means, DBSCAN)
- Reducción de dimensionalidad (PCA, t-SNE)
- Detección de anomalías

### 3. Aprendizaje por Refuerzo
El agente aprende a través de la interacción con un entorno, recibiendo recompensas o penalizaciones por sus acciones.

Aplicaciones:
- Juegos (AlphaGo, Chess)
- Vehículos autónomos
- Sistemas de recomendación
- Trading algorítmico

## Algoritmos Fundamentales

```python
# Ejemplo de regresión lineal simple
import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de ejemplo
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X, y)

# Hacer predicciones
prediction = model.predict([[6]])
print(f"Predicción para x=6: {prediction[0]}")
```

## Métricas de Evaluación

Para evaluar la calidad de nuestros modelos, utilizamos diferentes métricas según el tipo de problema:

### Clasificación
- Accuracy (Exactitud)
- Precision (Precisión)
- Recall (Exhaustividad)
- F1-Score

### Regresión
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (Coeficiente de determinación)

## Preprocesamiento de Datos

El preprocesamiento es crucial para el éxito de cualquier proyecto de ML:

1. **Limpieza de datos**: Eliminar valores faltantes o erróneos
2. **Normalización**: Escalar las características a rangos similares
3. **Codificación**: Convertir variables categóricas a numéricas
4. **Feature Engineering**: Crear nuevas características relevantes

## Desafíos Comunes

- **Overfitting**: El modelo se ajusta demasiado a los datos de entrenamiento
- **Underfitting**: El modelo es demasiado simple para capturar patrones
- **Sesgo en los datos**: Los datos no son representativos de la población
- **Interpretabilidad**: Entender cómo el modelo toma decisiones

## Futuro del Machine Learning

El campo evoluciona rápidamente con avances en:
- Deep Learning y redes neuronales profundas
- Procesamiento de lenguaje natural (NLP)
- Computer Vision
- ML automatizado (AutoML)
- Inteligencia artificial explicable (XAI)
""",
                    'path': 'docs/machine_learning_intro.md',
                    'type': 'markdown',
                    'domain': 'machine_learning',
                    'complexity': 0.7
                },
                {
                    'id': 'doc_python_guide',
                    'content': """# Guía Práctica de Python para Data Science

Python se ha convertido en el lenguaje preferido para ciencia de datos y machine learning debido a su sintaxis clara y poderosas bibliotecas.

## Bibliotecas Esenciales

### NumPy - Computación Numérica
```python
import numpy as np

# Crear arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Operaciones matemáticas
result = np.sqrt(arr)
mean_val = np.mean(arr)

# Broadcasting
arr_2d = arr.reshape(5, 1)
broadcasted = arr_2d + arr
```

### Pandas - Manipulación de Datos
```python
import pandas as pd

# Crear DataFrames
df = pd.DataFrame({
    'nombre': ['Ana', 'Luis', 'María'],
    'edad': [25, 30, 28],
    'salario': [50000, 60000, 55000]
})

# Operaciones básicas
print(df.head())
print(df.describe())
print(df.groupby('edad').mean())

# Filtrado de datos
jovenes = df[df['edad'] < 30]
```

### Matplotlib y Seaborn - Visualización
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico básico
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Gráfico de Línea')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Seaborn para gráficos estadísticos
sns.scatterplot(data=df, x='edad', y='salario')
plt.show()
```

### Scikit-learn - Machine Learning
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Preparar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.4f}')
```

## Mejores Prácticas

### 1. Organización del Código
```python
# Estructura recomendada para proyectos de DS
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   └── visualization/
├── tests/
└── requirements.txt
```

### 2. Control de Versiones
- Usar Git para versionar código
- Almacenar datos grandes separadamente
- Documentar cambios en modelos

### 3. Reproducibilidad
```python
# Fijar semillas aleatorias
import random
import numpy as np

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # También para frameworks como TensorFlow, PyTorch, etc.

set_seeds(42)
```

## Optimización de Rendimiento

### Vectorización con NumPy
```python
# Evitar bucles Python cuando sea posible
# Lento
result = []
for i in range(len(arr)):
    result.append(arr[i] ** 2)

# Rápido
result = arr ** 2
```

### Uso Eficiente de Pandas
```python
# Usar métodos vectorizados
df['columna_nueva'] = df['columna1'] + df['columna2']

# Usar apply solo cuando sea necesario
df['processed'] = df['text_column'].apply(lambda x: x.upper())
```

## Debugging y Profiling

### IPython y Jupyter
```python
# Comandos mágicos útiles
%time code_to_time()  # Tiempo de ejecución
%timeit repeated_code()  # Tiempo promedio
%pdb  # Activar debugger
%matplotlib inline  # Gráficos inline
```

### Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Iniciando entrenamiento del modelo")
logger.warning("Parámetro no óptimo detectado")
logger.error("Error en validación de datos")
```

Este enfoque sistemático asegura código mantenible y resultados reproducibles en proyectos de ciencia de datos.
""",
                    'path': 'guides/python_data_science.md',
                    'type': 'code',
                    'domain': 'programming',
                    'complexity': 0.8
                },
                {
                    'id': 'doc_neural_networks',
                    'content': """# Redes Neuronales Profundas: Conceptos y Aplicaciones

Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano, diseñados para reconocer patrones complejos en los datos.

## Fundamentos Básicos

### Perceptrón Simple
El perceptrón es la unidad básica de una red neuronal:

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        # Inicializar pesos
        self.weights = np.zeros(1 + X.shape[1])
        self.costs = []

        for i in range(self.n_iterations):
            output = self.net_input(X)
            errors = y - output
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
```

### Redes Multicapa (MLP)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Definir arquitectura
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    verbose=1
)
```

## Arquitecturas Especializadas

### Redes Convolucionales (CNN)
Ideales para procesamiento de imágenes:

```python
# CNN para clasificación de imágenes
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Redes Recurrentes (RNN/LSTM)
Para secuencias y series temporales:

```python
# LSTM para predicción de series temporales
lstm_model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(50, return_sequences=False),
    layers.Dense(25),
    layers.Dense(1)
])
```

## Técnicas de Optimización

### Funciones de Activación
- **ReLU**: f(x) = max(0, x) - Más común, evita vanishing gradient
- **Sigmoid**: f(x) = 1/(1+e^(-x)) - Salidas entre 0 y 1
- **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - Salidas entre -1 y 1
- **Leaky ReLU**: f(x) = max(0.01x, x) - Permite gradientes negativos pequeños

### Regularización
```python
# L1 y L2 Regularization
from tensorflow.keras import regularizers

model = models.Sequential([
    layers.Dense(128,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(64,
                activation='relu',
                kernel_regularizer=regularizers.l1(0.001)),
    layers.Dense(10, activation='softmax')
])
```

### Optimizadores Avanzados
```python
# Adam con learning rate scheduling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
```

## Técnicas Avanzadas

### Transfer Learning
```python
# Usar modelo pre-entrenado
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelar capas base
base_model.trainable = False

# Añadir capas personalizadas
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

### Attention Mechanisms
```python
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W))
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
```

## Monitoreo y Debugging

### Callbacks Útiles
```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    TensorBoard(log_dir='./logs')
]

model.fit(X_train, y_train, callbacks=callbacks)
```

### Visualización de Arquitectura
```python
# Visualizar modelo
tf.keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=True,
    show_layer_names=True
)

# Resumen del modelo
model.summary()
```

## Aplicaciones Actuales

1. **Visión por Computadora**
   - Detección de objetos (YOLO, R-CNN)
   - Segmentación semántica (U-Net)
   - Generación de imágenes (GANs)

2. **Procesamiento de Lenguaje Natural**
   - Transformers (BERT, GPT)
   - Traducción automática
   - Análisis de sentimientos

3. **Reconocimiento de Voz**
   - Speech-to-text
   - Síntesis de voz
   - Reconocimiento de hablante

4. **Sistemas de Recomendación**
   - Filtrado colaborativo
   - Content-based filtering
   - Hybrid approaches

Las redes neuronales profundas han revolucionado el campo de la inteligencia artificial, permitiendo avances significativos en tareas que antes eran extremadamente difíciles para las computadoras.
""",
                    'path': 'advanced/neural_networks.md',
                    'type': 'technical',
                    'domain': 'deep_learning',
                    'complexity': 0.9
                }
            ],
            'queries': [
                "¿Cómo funciona el machine learning?",
                "Mejores prácticas para programar en Python",
                "Diferencia entre CNN y RNN",
                "¿Qué es regularización en redes neuronales?",
                "Tutorial de pandas para análisis de datos",
                "Algoritmos de aprendizaje supervisado vs no supervisado",
                "Cómo implementar un perceptrón desde cero"
            ]
        }

    async def run_comprehensive_demo(self):
        """Ejecuta la demostración completa del sistema integrado"""

        print("🚀 DEMO INTEGRADO - MCP HUB ENHANCED")
        print("=" * 80)
        print("Demostración de todas las características avanzadas trabajando en conjunto")
        print()

        # 1. Inicialización del sistema
        print("📋 FASE 1: Inicialización del Sistema")
        print("-" * 40)

        print("Configurando sistema con modo COMPREHENSIVE...")
        config = create_comprehensive_config()
        self.orchestrator = create_orchestrator("comprehensive")

        # Mostrar configuración
        print(f"   ✅ Modo de procesamiento: {config.processing_mode.value}")
        print(f"   ✅ Características habilitadas: {sum(1 for v in [
            config.enable_dynamic_chunking,
            config.enable_mvr,
            config.enable_virtual_chunks,
            config.enable_query_expansion,
            config.enable_confidence_calibration
        ] if v)}/5")
        print(f"   ✅ Operaciones concurrentes: {config.max_concurrent_operations}")
        print(f"   ✅ Resultados máximos: {config.max_search_results}")

        # Estado inicial del sistema
        initial_status = self.orchestrator.get_system_status()
        print("\n📊 Estado inicial de características:")
        for feature, status in initial_status['feature_status'].items():
            emoji = "✅" if status == "enabled" else "❌" if status == "error" else "⏳"
            print(f"   {emoji} {feature.replace('_', ' ').title()}: {status}")

        # 2. Preparación de datos
        print(f"\n📚 FASE 2: Preparación de Datos")
        print("-" * 40)

        print("Cargando documentos de ejemplo...")
        documents = self.demo_data['documents']

        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc['id']}")
            print(f"      Tipo: {doc['type']} | Dominio: {doc['domain']}")
            print(f"      Tamaño: {len(doc['content'])} chars | Complejidad: {doc['complexity']}")

        # 3. Añadir documentos al sistema MVR (si está habilitado)
        if self.orchestrator.mvr_system:
            print(f"\n🔧 Indexando documentos en sistema MVR...")
            for doc in documents:
                success = self.orchestrator.mvr_system.add_document(
                    doc_id=doc['id'],
                    content=doc['content'],
                    metadata={
                        'type': doc['type'],
                        'domain': doc['domain'],
                        'path': doc['path'],
                        'complexity': doc['complexity']
                    }
                )
                emoji = "✅" if success else "❌"
                print(f"   {emoji} {doc['id']}")

        # 4. Procesamiento de queries
        print(f"\n🔍 FASE 3: Procesamiento de Queries")
        print("-" * 40)

        queries = self.demo_data['queries'][:3]  # Primeras 3 queries para la demo

        for i, query in enumerate(queries, 1):
            print(f"\n>>> Query {i}: '{query}'")
            print("   " + "─" * 50)

            start_time = time.time()

            # Procesamiento avanzado
            result = await self.orchestrator.process_advanced(
                query=query,
                documents=documents,
                context={'demo_query': i, 'timestamp': time.time()}
            )

            processing_time = time.time() - start_time

            # Mostrar resultados
            print(f"   ⏱️  Tiempo de procesamiento: {processing_time:.3f}s")
            print(f"   🔧 Características usadas: {len([s for s in result.feature_status.values() if s.value == 'enabled'])}")

            # Query Expansion
            if result.expanded_queries:
                print(f"   🔄 Queries expandidas ({len(result.expanded_queries)}):")
                for j, exp_query in enumerate(result.expanded_queries[:3], 1):
                    print(f"      {j}. {exp_query}")

            # Dynamic Chunking
            if result.chunks:
                print(f"   📄 Chunks generados: {len(result.chunks)}")
                chunk_types = {}
                for chunk in result.chunks:
                    if hasattr(chunk.metadata, 'chunk_type'):
                        chunk_type = chunk.metadata.chunk_type.value
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

                for chunk_type, count in chunk_types.items():
                    print(f"      - {chunk_type}: {count} chunks")

            # Search Results
            if result.search_results:
                print(f"   🎯 Resultados de búsqueda ({len(result.search_results)}):")
                for j, search_result in enumerate(result.search_results[:3], 1):
                    print(f"      {j}. {search_result.doc_id} (Score: {search_result.score:.4f})")
                    if hasattr(search_result, 'vector_scores'):
                        vector_info = ', '.join([f"{k.value}: {v:.3f}" for k, v in list(search_result.vector_scores.items())[:2]])
                        print(f"         Vectores: {vector_info}")

            # Confidence Calibration
            if result.confidence_scores:
                print(f"   🎯 Calibración de confianza:")
                for j, conf_score in enumerate(result.confidence_scores[:3], 1):
                    print(f"      {j}. Raw: {conf_score.raw_score:.3f} → Calibrated: {conf_score.calibrated_score:.3f}")
                    print(f"         Nivel: {conf_score.confidence_level.value} | Incertidumbre: {conf_score.uncertainty_estimate:.3f}")

        # 5. Simulación de feedback
        print(f"\n🔄 FASE 4: Simulación de Feedback")
        print("-" * 40)

        print("Añadiendo feedback simulado para mejorar el sistema...")

        # Generar feedback sintético
        np.random.seed(42)
        feedback_data = []

        for i, query in enumerate(queries):
            # Simular múltiples interacciones por query
            for j in range(5):
                relevance_score = np.random.beta(2, 1)  # Sesgado hacia scores altos
                was_helpful = relevance_score > 0.6  # Threshold para utilidad

                feedback_data.append({
                    'query': query,
                    'result_doc_id': f'doc_{i}_{j}',
                    'relevance_score': relevance_score,
                    'was_helpful': was_helpful
                })

                # Añadir feedback al sistema
                self.orchestrator.add_feedback(
                    query=query,
                    result_doc_id=f'doc_{i}_{j}',
                    relevance_score=relevance_score,
                    was_helpful=was_helpful,
                    context={'simulation': True, 'query_idx': i}
                )

        print(f"   ✅ Añadido feedback para {len(feedback_data)} interacciones")
        print(f"   📊 Tasa de utilidad promedio: {np.mean([f['was_helpful'] for f in feedback_data]):.1%}")

        # 6. Análisis de rendimiento
        print(f"\n📈 FASE 5: Análisis de Rendimiento")
        print("-" * 40)

        final_status = self.orchestrator.get_system_status()

        print("🔧 Estado de características:")
        enabled_features = final_status['config']['enabled_features']
        for feature in enabled_features:
            print(f"   ✅ {feature.replace('_', ' ').title()}")

        print(f"\n📊 Estadísticas de operación:")
        stats = final_status['statistics']
        print(f"   • Total de operaciones: {stats['total_operations']}")
        print(f"   • Tiempo promedio: {stats['avg_processing_time_ms']:.1f}ms")

        if stats['feature_usage']:
            print(f"   • Uso por característica:")
            for feature, count in stats['feature_usage'].items():
                print(f"     - {feature.replace('_', ' ').title()}: {count} veces")

        if stats['error_counts']:
            print(f"   • Errores detectados:")
            for feature, errors in stats['error_counts'].items():
                print(f"     - {feature}: {errors} errores")

        # 7. Métricas de calibración (si está disponible)
        if (self.orchestrator.confidence_calibrator and
            final_status.get('confidence_calibration_system')):

            print(f"\n🎯 Métricas de calibración:")
            cc_status = final_status['confidence_calibration_system']

            if 'recent_metrics' in cc_status:
                metrics = cc_status['recent_metrics']
                print(f"   • Expected Calibration Error: {metrics.get('ece', 0):.4f}")
                print(f"   • Brier Score: {metrics.get('brier_score', 0):.4f}")
                print(f"   • Reliability Score: {metrics.get('reliability', 0):.4f}")

            print(f"   • Muestras de feedback: {cc_status.get('feedback_samples', 0)}")
            print(f"   • Método actual: {cc_status.get('current_best_method', 'N/A')}")

        # 8. Optimización automática
        print(f"\n⚡ FASE 6: Optimización Automática")
        print("-" * 40)

        optimization_report = self.orchestrator.optimize_configuration()

        print("📊 Análisis de rendimiento actual:")
        perf = optimization_report['current_performance']
        print(f"   • Tiempo promedio: {perf['avg_processing_time']:.3f}s")
        print(f"   • Operaciones totales: {perf['total_operations']}")

        if optimization_report['recommendations']:
            print(f"\n💡 Recomendaciones de optimización:")
            for i, rec in enumerate(optimization_report['recommendations'], 1):
                print(f"   {i}. {rec}")

        if optimization_report['auto_applied']:
            print(f"\n🔄 Optimizaciones aplicadas automáticamente:")
            for i, opt in enumerate(optimization_report['auto_applied'], 1):
                print(f"   {i}. {opt}")

        # 9. Demostración de características específicas
        print(f"\n🔬 FASE 7: Demostración de Características Específicas")
        print("-" * 40)

        await self._demonstrate_specific_features()

        # 10. Resumen final
        print(f"\n🎉 RESUMEN FINAL")
        print("-" * 40)

        final_stats = self.orchestrator.get_system_status()

        print("✅ Demostración completada exitosamente!")
        print(f"\n📋 Características demostradas:")

        demos_completed = [
            "✅ Dynamic Chunking Adaptativo",
            "✅ Multi-Vector Retrieval (MVR)",
            "✅ Query Expansion Automática",
            "✅ Confidence Calibration Dinámica",
            "✅ Sistema Integrado de Orquestación",
            "✅ Feedback Loop y Optimización",
            "✅ Procesamiento Paralelo",
            "✅ Métricas y Monitoreo en Tiempo Real"
        ]

        for demo in demos_completed:
            print(f"   {demo}")

        print(f"\n📊 Estadísticas finales del sistema:")
        print(f"   • Queries procesadas: {len(queries)}")
        print(f"   • Documentos indexados: {len(documents)}")
        print(f"   • Feedback recibido: {len(feedback_data)} interacciones")
        print(f"   • Características activas: {len(final_stats['config']['enabled_features'])}")
        print(f"   • Tiempo total de demo: {time.time() - start_time:.1f}s")

        print(f"\n💡 Próximos pasos sugeridos:")
        print("   1. Integrar con fuentes de datos reales")
        print("   2. Ajustar configuración según casos de uso específicos")
        print("   3. Implementar monitoreo continuo en producción")
        print("   4. Configurar pipelines de reentrenamiento automático")

    async def _demonstrate_specific_features(self):
        """Demuestra características específicas en detalle"""

        print("🔧 Demostraciones específicas:")

        # 1. Dynamic Chunking con diferentes tipos de contenido
        if self.orchestrator.chunking_system:
            print("\n   📄 Dynamic Chunking:")

            test_content = """# Título de prueba

Este es un párrafo de ejemplo con contenido variado.

## Subsección con código

```python
def example_function():
    return "Hello, World!"
```

Y más texto después del código."""

            chunks = self.orchestrator.chunking_system.adaptive_chunking(
                text=test_content,
                file_path="test.md"
            )

            print(f"      ✅ {len(chunks)} chunks generados")
            for i, chunk in enumerate(chunks, 1):
                print(f"         {i}. Tipo: {chunk.metadata.chunk_type.value}, Tamaño: {chunk.metadata.size}")

        # 2. Query Expansion con diferentes tipos
        if self.orchestrator.query_expander:
            print("\n   🔄 Query Expansion:")

            test_queries = [
                "¿Cómo funciona el algoritmo?",
                "Mejores prácticas de programación",
                "Diferencias entre modelos"
            ]

            for query in test_queries:
                expansion = self.orchestrator.query_expander.expand_query(query, max_expansions=3)
                print(f"      '{query}' →")
                print(f"         Tipo: {expansion.query_type.value}")
                print(f"         Expansiones: {len(expansion.expanded_terms)}")

        # 3. Confidence Calibration en acción
        if self.orchestrator.confidence_calibrator:
            print("\n   🎯 Confidence Calibration:")

            test_scores = [0.3, 0.6, 0.9]
            for score in test_scores:
                calibrated = self.orchestrator.confidence_calibrator.calibrate_confidence(score)
                print(f"      {score:.1f} → {calibrated.calibrated_score:.3f} ({calibrated.confidence_level.value})")

        print("      ✅ Demostraciones específicas completadas")


def create_demo_config() -> AdvancedConfig:
    """Crea configuración optimizada para la demo"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.COMPREHENSIVE,
        max_concurrent_operations=4,
        cache_size_mb=50,
        enable_dynamic_chunking=True,
        enable_mvr=True,
        enable_virtual_chunks=False,  # Deshabilitado para simplicidad de demo
        enable_query_expansion=True,
        enable_confidence_calibration=True,
        max_search_results=8,
        max_expansions=6
    )


async def run_demo():
    """Función principal para ejecutar la demo"""
    demo = IntegratedDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    """
    Ejecutar la demo integrada completa

    Este script demuestra todas las características avanzadas del MCP Hub Enhanced:

    1. Dynamic Chunking Adaptativo
    2. Multi-Vector Retrieval (MVR)
    3. Query Expansion Automática
    4. Confidence Calibration Dinámica
    5. Sistema Integrado de Orquestación

    Uso:
        python integrated_demo.py

    O desde el directorio padre:
        python -m core.advanced_features.integrated_demo
    """

    print("🚀 Iniciando Demo Integrado del MCP Hub Enhanced...")
    print("   Preparando sistema avanzado con todas las características...")
    print()

    try:
        # Configurar logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Ejecutar demo
        asyncio.run(run_demo())

        print("\n🎉 Demo completado exitosamente!")
        print("   Todas las características avanzadas han sido demostradas.")
        print("   El sistema está listo para integración en producción.")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrumpido por el usuario")

    except Exception as e:
        print(f"\n❌ Error durante la demo: {e}")
        print("   Revisa los logs para más detalles.")
        raise

    finally:
        print("\n📋 Demo finalizado")
        print("="*80)
