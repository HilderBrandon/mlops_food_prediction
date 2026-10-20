import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(input_path: str, output_path: str):
    print(f"Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path)
    
    # Inspección rápida de valores nulos o duplicados
    print(f"Registros originales: {df.shape[0]}, Columnas: {df.shape[1]}")
    
    # Codificación de variables categóricas para modelos de Machine Learning
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar datos procesados
    df.to_csv(output_path, index=False)
    print(f"Datos limpios guardados exitosamente en: {output_path}")

if __name__ == "__main__":
    # Ajusta el nombre del archivo según el CSV exacto que descargaste de Kaggle
    input_file = "data/raw/online food delivery dataset.csv" 
    output_file = "data/processed/cleaned_food_data.csv"
    
    preprocess_data(input_file, output_file)