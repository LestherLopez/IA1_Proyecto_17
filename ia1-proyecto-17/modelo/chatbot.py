# -*- coding: utf-8 -*-
import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk
import sys
import io
from tensorflow.keras.models import load_model

# Forzar la salida en UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Asegúrate de haber descargado las dependencias de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo entrenado y los archivos de palabras y clases
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Cargar las intenciones desde el archivo JSON
intents = json.loads(open('intents.json', encoding='utf-8').read())

# Inicializar el lematizador
lemmatizer = WordNetLemmatizer()

# Función para obtener la bolsa de palabras (bag of words)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

# Función para convertir la frase en un vector de características (bag of words)
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Función para predecir la clase de una frase
def predict_class(sentence):
    # Crear una bolsa de palabras para la entrada
    p = bow(sentence, words)
    # Obtener la predicción
    pred = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    # Obtener el índice de la clase con la probabilidad más alta
    pred_idx = np.argmax(pred)
    # Si la probabilidad es mayor que el umbral de error, devuelve la clase
    if pred[pred_idx] > ERROR_THRESHOLD:
        return classes[pred_idx]
    else:
        return None

# Función para obtener una respuesta basada en la clase predicha
def get_response(intent_tag):
    # Buscar la intención correspondiente a la etiqueta
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            return response

# Función para procesar la entrada del usuario
def chatbot_response(msg):
    # Predecir la clase
    intent = predict_class(msg)
    # Obtener una respuesta en función de la clase predicha
    if intent:
        response = get_response(intent)
    else:
        response = "Lo siento, no te entiendo. ¿Puedes reformular?"
    return response

# Función para sanitizar la salida
def sanitize_output(text):
    return text.encode('utf-8').decode('utf-8')

# Interacción con el chatbot
print("GO! Bot is running!")
while True:
    # Obtener la entrada del usuario
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: ¡Adiós!")
        break
    # Obtener la respuesta del chatbot y mostrarla
    response = chatbot_response(message)
    print("Bot:", sanitize_output(response))
