Español:

Este código realiza el entrenamiento y la evaluación de un modelo de predicción para precios de criptomonedas utilizando redes neuronales profundas. A continuación, se detallan las principales funciones y componentes utilizados:

Conexión a la base de datos: Utiliza psycopg2 para conectarse a una base de datos PostgreSQL, donde recupera datos históricos de precios de criptomonedas, ordenados por fecha. La información es almacenada en un DataFrame para su posterior procesamiento.

Preprocesamiento de los datos: La fecha de cada registro se convierte en el índice temporal. Luego, se escalan los valores de las características relevantes usando MaxAbsScaler, lo que asegura que los datos estén dentro del mismo rango, facilitando el aprendizaje del modelo.

Creación de conjuntos de entrenamiento: Se genera un conjunto de datos con una ventana temporal definida (llamada time_step) para predecir el precio de cierre (asumido como la tercera columna). La función create_dataset organiza los datos en secuencias de entrada y valores de salida.

Evaluación del modelo: Se utiliza la métrica sMAPE (Error Porcentual Absoluto Simétrico) para evaluar la precisión de las predicciones del modelo, así como otras métricas comunes como el error cuadrático medio (MSE), el error absoluto medio (MAE) y el coeficiente de determinación (R²).

Modelo de redes neuronales: Se construye un modelo de red neuronal con varias capas: una capa Conv1D para la extracción de características, una capa LSTM para la memoria temporal y una capa de atención MultiHeadAttention para mejorar la capacidad de la red de enfocarse en diferentes partes de la secuencia. Además, se utiliza regularización L2 en varias capas y un decaimiento de L2 durante el entrenamiento para evitar el sobreajuste.

Optimización de hiperparámetros: Se utiliza Keras Tuner con optimización bayesiana para encontrar la mejor configuración de hiperparámetros del modelo (como el número de filtros en la capa convolucional, el número de unidades en las capas LSTM, la tasa de aprendizaje, etc.).

Entrenamiento y evaluación: Se entrena el modelo utilizando el conjunto de entrenamiento, se evalúa su rendimiento en un conjunto de validación y se almacenan las métricas de error. Los mejores modelos se guardan para su posterior uso.

Almacenamiento de resultados: Las métricas de error se almacenan en una base de datos para poder analizarlas más adelante. Esto permite la comparación del rendimiento de distintos modelos.

Callback de regularización L2: Un callback personalizado ajusta la regularización L2 de las capas específicas al final de cada época de entrenamiento para reducir el sobreajuste.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
English:
This code performs training and evaluation of a prediction model for cryptocurrency prices using deep neural networks. Below are the main functions and components used:

Database Connection: It uses psycopg2 to connect to a PostgreSQL database, retrieving historical cryptocurrency price data, sorted by date. The data is stored in a DataFrame for further processing.

Data Preprocessing: The date of each record is converted to a time index. Then, relevant feature values are scaled using MaxAbsScaler, ensuring that the data is within the same range, which aids the model's learning process.

Training Dataset Creation: A dataset is generated with a defined time window (time_step) to predict the closing price (assumed to be the third column). The create_dataset function organizes the data into input sequences and output values.

Model Evaluation: The sMAPE (Symmetric Mean Absolute Percentage Error) metric is used to evaluate the accuracy of the model's predictions, along with other common metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².

Neural Network Model: A neural network model is built with several layers: a Conv1D layer for feature extraction, an LSTM layer for temporal memory, and a MultiHeadAttention layer to enhance the network's ability to focus on different parts of the sequence. L2 regularization is applied to several layers, and L2 decay during training is used to prevent overfitting.

Hyperparameter Optimization: Keras Tuner is used with Bayesian Optimization to find the best hyperparameter configuration for the model (such as the number of filters in the convolutional layer, the number of units in the LSTM layers, the learning rate, etc.).

Training and Evaluation: The model is trained using the training set, evaluated on a validation set, and error metrics are stored. The best models are saved for future use.

Result Storage: Error metrics are stored in a database for later analysis. This allows comparison of the performance of different models.

L2 Regularization Callback: A custom callback adjusts the L2 regularization of specific layers at the end of each training epoch to reduce overfitting.
