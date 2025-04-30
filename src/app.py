from utils import db_connect
engine = db_connect()

# your code here

# Importación de librerías necesarias
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  

# Paso 1: Carga los datos desde la URL proporcionada
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# Verificamos las columnas disponibles en el dataset
print("Columnas disponibles en el dataset:", df.columns)

# Paso 2: Separa las variables independientes (X) del objetivo (y)
target_column = "quality"
X = df.drop(columns=[target_column])  
y = df[target_column]  

# Verificar las etiquetas únicas
print("Etiquetas de calidad presentes en el dataset:", y.unique())

# Paso 3: Divide el conjunto de datos en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Escala los datos para mejorar el rendimiento del modelo KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Paso 5: Entrena el modelo KNN con un valor inicial de k
k_initial = 5
knn_model = KNeighborsClassifier(n_neighbors=k_initial)
knn_model.fit(X_train_scaled, y_train)

# Paso 6: Evalúa el rendimiento del modelo inicial
y_pred = knn_model.predict(X_test_scaled)
print(f"Exactitud inicial con k={k_initial}: {accuracy_score(y_test, y_pred):.4f}")
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Paso 7: Optimización del valor de k probando diferentes opciones
k_values = range(1, 21)
accuracy_results = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    accuracy_results.append(accuracy_score(y_test, y_pred_temp))

# Paso 8: Grafica accuracy vs k para encontrar el mejor valor
plt.plot(k_values, accuracy_results, marker="o")
plt.xlabel("Número de Vecinos (k)")
plt.ylabel("Exactitud")
plt.title("Optimización del parámetro k en KNN")
plt.show()

# Paso 9: Crea una función que reciba valores numéricos y prediga la calidad del vino
def predict_wine_quality(features):
    features_scaled = scaler.transform([features])  
    prediction = knn_model.predict(features_scaled)[0]  
    
    # Diccionario actualizado con todas las etiquetas posibles
    quality_labels = {
        3: "Muy baja calidad 🍷", 4: "Baja calidad 🍷", 5: "Calidad media 🍷",
        6: "Buena calidad 🍷", 7: "Alta calidad 🍷", 8: "Excelente calidad 🍷"
    }
    
    return f"Este vino probablemente sea de {quality_labels.get(prediction, 'calidad desconocida')}"

# Ejemplo de uso de la función de predicción
sample_wine = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
print(predict_wine_quality(sample_wine))
