import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import joblib

def train_pipeline(data_path: str, model_output_path: str):
    print(f"Cargando datos procesados desde: {data_path}")
    df = pd.read_csv(data_path)
    
    # Identificar la columna objetivo (Output / Preferencia de pedido)
    target_col = 'Output' if 'Output' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Dividir datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entrenando el modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el rendimiento
    predictions = model.predict(X_test)
    print("\n--- Reporte de Evaluación del Modelo ---")
    print(classification_report(y_test, predictions))
    
    # Guardar el artefacto del modelo en la carpeta models/
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModelo guardado exitosamente en: {model_output_path}")

if __name__ == "__main__":
    input_data = "data/processed/cleaned_food_data.csv"
    output_model = "models/model_v1.pkl"
    
    train_pipeline(input_data, output_model)