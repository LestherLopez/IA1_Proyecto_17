import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Cargar y procesar el archivo de texto
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"¿qué tal?", "qué tal", txt)
    txt = re.sub(r"¡hola!", "hola", txt)
    txt = re.sub(r"de acuerdo", "está bien", txt)
    txt = re.sub(r"[^\w\s]", "", txt)  # Eliminar caracteres no alfabéticos
    return txt

questions = []
answers = []

with open('greetings_spanish.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):  # Asumimos que cada par de líneas es pregunta-respuesta
        question = clean_text(lines[i].strip())
        
        # Aseguramos que no se intente acceder a una línea fuera de rango
        if i + 1 < len(lines):
            answer = clean_text(lines[i + 1].strip())
            questions.append(question)
            answers.append(answer)
        else:
            print(f"Advertencia: línea sin respuesta para la pregunta: '{question}'")

# Añadir tokens especiales al vocabulario
special_tokens = ['<SOS>', '<EOS>', '<OUT>']
tokenizer = Tokenizer(filters='', lower=True, oov_token='<OUT>')
tokenizer.fit_on_texts(questions + answers + special_tokens)  # Incluir los tokens especiales

# Asegurarnos de que los tokens especiales estén en el vocabulario
for token in special_tokens:
    if token not in tokenizer.word_index:
        tokenizer.word_index[token] = len(tokenizer.word_index) + 1

vocab_size = len(tokenizer.word_index) + 1  # Tamaño del vocabulario

# Convertir preguntas y respuestas a secuencias numéricas
encoder_inp = tokenizer.texts_to_sequences(questions)
decoder_inp = tokenizer.texts_to_sequences(answers)

# Padding de las secuencias
max_sequence_length = max([len(seq) for seq in encoder_inp + decoder_inp])
encoder_inp = pad_sequences(encoder_inp, maxlen=max_sequence_length, padding='post')
decoder_inp = pad_sequences(decoder_inp, maxlen=max_sequence_length, padding='post')

# Definir las salidas decodificadas (para el entrenamiento)
decoder_final_output = np.zeros((len(answers), max_sequence_length, vocab_size))
for i, seq in enumerate(decoder_inp):
    for t, word_idx in enumerate(seq):
        if word_idx != 0:
            decoder_final_output[i, t, word_idx] = 1.0

# Crear el modelo de secuencia a secuencia
latent_dim = 256  # Tamaño de la capa LSTM

# Codificador
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decodificador
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_concat_input = decoder_dense(decoder_outputs)

# Modelo de entrenamiento
model = Model([encoder_inputs, decoder_inputs], decoder_concat_input)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=10)

# Preparar el modelo para inferencia (predicción)
enc_model = Model(encoder_inputs, encoder_states)

# Decodificador (para predicción)
dec_state_input_h = Input(shape=(latent_dim,))
dec_state_input_c = Input(shape=(latent_dim,))
dec_state_inputs = [dec_state_input_h, dec_state_input_c]

dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_lstm_outputs, dec_state_h, dec_state_c = dec_lstm(decoder_embedding, initial_state=dec_state_inputs)
dec_dense = Dense(vocab_size, activation='softmax')
dec_outputs = dec_dense(dec_lstm_outputs)

dec_model = Model([decoder_inputs] + dec_state_inputs, [dec_outputs, dec_state_h, dec_state_c])

# Crear el diccionario inverso para obtener palabras de los índices
inv_vocab = {idx: word for word, idx in tokenizer.word_index.items()}

# Función para interactuar con el chatbot
def chat():
    print("##########################################")
    print("#       start chatting ver. 1.0          #")
    print("##########################################")
    
    prepro1 = ""
    while prepro1 != 'q':
        prepro1 = input("tú: ")
        prepro1 = clean_text(prepro1)
        prepro = [prepro1]
        
        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                try:
                    lst.append(tokenizer.word_index[y])
                except:
                    lst.append(tokenizer.word_index['<OUT>'])
            txt.append(lst)

        txt = pad_sequences(txt, maxlen=max_sequence_length, padding='post')
        stat = enc_model.predict(txt)

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['<SOS>']  # Ahora el token <SOS> está en el vocabulario

        stop_condition = False
        decoded_translation = ''

        while not stop_condition:
            dec_outputs, dec_state_h, dec_state_c = dec_model.predict([empty_target_seq] + stat)
            decoder_concat_input = dec_outputs

            sample_word_index = np.argmax(decoder_concat_input[0, -1, :])
            sampled_word = inv_vocab.get(sample_word_index, '') + ' '

            if sampled_word != '<EOS>':
                decoded_translation += sampled_word

            if sampled_word == '<EOS>' or len(decoded_translation.split()) > 13:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sample_word_index
            stat = [dec_state_h, dec_state_c]

        print("chatbot: ", decoded_translation)

# Ejecutar el chatbot
chat()
