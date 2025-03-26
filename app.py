import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Título de la aplicación
st.title("Asistente Cardiaco")
st.markdown("**Autor**: Alfredo Díaz Claro")

# Instrucciones de uso
st.markdown("""
### Instrucciones:
1. Usa los sliders para ingresar tu edad y el nivel de colesterol.
2. La aplicación te dirá si tienes o no problemas cardíacos basado en tu edad y nivel de colesterol.
3. Los datos son normalizados antes de hacer la predicción usando un modelo de clasificación entrenado con KNN.
""")

# Cargar el modelo de clasificación y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('esclador.bin')

# Crear pestañas para los inputs y resultados
tab1, tab2 = st.tabs(["Ingresar Datos", "Resultados"])

with tab1:
    # Definir los sliders
    edad = st.slider('Selecciona la edad', 18, 80, 25)  # Min = 18, Max = 80, Valor por defecto = 25
    colesterol = st.slider('Selecciona el nivel de colesterol', 100, 600, 200)  # Min = 100, Max = 600, Valor por defecto = 200
    
    # Crear un dataframe con los datos ingresados
    data = {
        'edad': [edad],
        'colesterol': [colesterol]
    }
    
    df = pd.DataFrame(data)
    st.write("DataFrame generado con los valores ingresados:")
    st.dataframe(df)

with tab2:
    # Normalización de los datos con MinMaxScaler
    df_normalizado = pd.DataFrame(escalador.transform(df), columns=df.columns)
    
    # Realizar la predicción con el modelo KNN
    prediccion = modelo_knn.predict(df_normalizado)
    
    # Mostrar el resultado de la predicción
    if prediccion == 1:
        st.write("**¡Tienes problemas cardíacos!**")
        st.image("https://cloudfront-us-east-1.images.arcpublishing.com/infobae/WRI4UH2CFFG3PFSLDLXBXW4YV4.jpg", caption="Problemas cardíacos")
    else:
        st.write("**¡No tienes problemas cardíacos!**")
        st.image("https://colombianadetrasplantes.com/web/wp-content/uploads/2023/05/01-PORTADA.-01-scaled.jpg", caption="Salud cardíaca")
