import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

# Configuración de la interfaz
st.set_page_config(page_title="Clasificación de Reseñas de IMDB usando CNN", layout="wide")

# Estilos CSS para la portada
st.markdown("""
    <style>
    .main {
        background-color: #f7f2f2;  
    }
    .title {
        color: #5a5a5a;  
        font-size: 30px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #8b5b93;  
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Título y subtítulo de la portada
st.markdown('<div class="title">UNIVERSIDAD NACIONAL DEL ALTIPLANO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NELIDA ARACELY QUISPE CALATAYUD</div>', unsafe_allow_html=True)

# Mostrar las imágenes
col1, col2 = st.columns([1, 1])
with col1:
    st.image("Unap.png", width=80)
with col2:
    st.image("Finesi.png", width=80)

# Crear espacio para que la portada sea más visible
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Configuración de la interfaz para el modelo de clasificación de reseñas de IMDB
st.title("Clasificación de Reseñas de IMDB usando CNN")

# Parámetros del modelo
st.sidebar.subheader("Parámetros del modelo")
vocab_size = st.sidebar.slider("Tamaño del vocabulario", 1000, 20000, 10000)
max_len = st.sidebar.slider("Longitud máxima de secuencia", 50, 500, 100)
embedding_dim = st.sidebar.slider("Dimensión de embedding", 8, 128, 50)
filters = st.sidebar.slider("Número de filtros de convolución", 16, 256, 128)
kernel_size = st.sidebar.slider("Tamaño del kernel", 3, 7, 5)

# Cargar y preprocesar el dataset de IMDB
st.write("Cargando el dataset de IMDB...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Rellenar las secuencias para que tengan la misma longitud
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Crear un Tokenizer basado en el vocabulario del dataset de IMDB
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(imdb.get_word_index())

# Construcción del modelo CNN
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
st.write("Modelo CNN construido para clasificación de texto.")

# Verificar si el estado de entrenamiento está en st.session_state
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Entrenamiento del modelo
if st.button("Entrenar modelo"):
    with st.spinner("Entrenando el modelo, por favor espera..."):
        # Entrenar el modelo
        history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))
        st.success("Modelo entrenado")
        # Marcar el modelo como entrenado
        st.session_state.trained = True  # Actualiza el estado a True después del entrenamiento

        # Mostrar métricas de rendimiento después del entrenamiento
        st.write("Evaluación del modelo en conjunto de prueba:")
        loss, accuracy = model.evaluate(x_test, y_test)
        st.write(f"Pérdida: {loss}")
        st.write(f"Precisión: {accuracy}")

# Entrada para probar el modelo con una reseña personalizada
st.write("Introduce una reseña para probar el modelo:")
review = st.text_area("Reseña:")

# Predicción de la reseña
if review and st.session_state.trained:  # Verificar si el modelo fue entrenado
    # Tokenizar y predecir la reseña
    review_sequence = tokenizer.texts_to_sequences([review])
    review_sequence = pad_sequences(review_sequence, maxlen=max_len)
    prediction = model.predict(review_sequence)
    label = "Positivo" if prediction[0] > 0.5 else "Negativo"
    st.write(f"La predicción del modelo es: {label}")
elif review:
    st.warning("El modelo aún no ha sido entrenado. Haz clic en 'Entrenar modelo' primero.")
