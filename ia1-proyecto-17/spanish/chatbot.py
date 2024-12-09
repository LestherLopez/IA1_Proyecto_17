import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import yaml

# Cargar los datos de entrenamiento desde un archivo YAML
with open('chatbot_data.yml', 'r') as f:
    data = yaml.safe_load(f)
    conversations = data['conversations'][0]['pairs']

# Preparar los textos de entrada y salida
input_texts = []
target_texts = []
input_lang = set()
target_lang = set()
for pair in conversations:
    input_text = pair['input']
    target_text = pair['output']
    
    # Añadir texto a las listas
    input_texts.append(input_text)
    target_texts.append(target_text)
    
    # Añadir caracteres únicos a los sets
    for char in input_text:
        input_lang.add(char)
    for char in target_text:
        target_lang.add(char)

# Añadir los tokens especiales
input_lang.add('<unk>')
input_lang.add('<pad>')
target_lang.add('\t')  # Start-of-sequence token
target_lang.add('\n')  # End-of-sequence token

# Crear los índices de los caracteres
input_token_index = {char: i for i, char in enumerate(sorted(input_lang))}
target_token_index = {char: i for i, char in enumerate(sorted(target_lang))}
reverse_input_char_index = {i: char for char, i in input_token_index.items()}
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

# Configuraciones del modelo
max_encoder_seq_length = max(len(txt) for txt in input_texts)
max_decoder_seq_length = max(len(txt) for txt in target_texts)
num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)

# Guardar las configuraciones para usarlas más tarde
with open('chatbot_data.pkl', 'wb') as f:
    pickle.dump([input_texts, target_texts, input_lang, target_lang, input_token_index, target_token_index, 
                 max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, 
                 reverse_input_char_index, reverse_target_char_index], f)

# Crear el modelo de secuencia a secuencia (Encoder-Decoder)
def create_model(num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length):
    # Encoder
    encoder_inputs = Input(shape=(None,))  # Input de la secuencia, longitud variable
    encoder_embedding = Embedding(num_encoder_tokens, 256)(encoder_inputs)  # Output (batch_size, timesteps, 256)
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(None,))  # Input de la secuencia de salida
    decoder_embedding = Embedding(num_decoder_tokens, 256)(decoder_inputs)  # Output (batch_size, timesteps, 256)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Modelo
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# Crear el modelo
model = create_model(num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length)

# Compilar el modelo
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparar los datos de entrada para el entrenamiento
def train_model():
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t] = input_token_index[char]
        for t, char in enumerate(target_text):
            decoder_input_data[i, t] = target_token_index[char]
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    
    # Expandir las dimensiones de las entradas para que tengan la forma correcta para LSTM
    encoder_input_data = np.expand_dims(encoder_input_data, -1)  # Esto agrega la dimensión de "features" (dim=1)
    decoder_input_data = np.expand_dims(decoder_input_data, -1)  # Esto también agrega la dimensión de "features"
    
    # Entrenar el modelo
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=10, validation_split=0.2)
    model.save('chatbot_model.h5')

# Si el modelo no está entrenado, entrenarlo
try:
    model = tf.keras.models.load_model('chatbot_model.h5')
    print("Modelo cargado con éxito.")
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
except:
    print("Modelo no encontrado, entrenando el modelo.")
    train_model()

# Función para codificar las secuencias de entrada
def encode_input_text(input_text):
    input_seq = [input_token_index.get(char, input_token_index['<unk>']) for char in input_text]
    return pad_sequences([input_seq], maxlen=max_encoder_seq_length, padding='post')

# Función para predecir la secuencia
def predict_sequence(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array([target_token_index['\t']])  # Token de inicio

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.array([sampled_token_index])
        states_value = [h, c]

    return decoded_sentence

# Cargar los modelos de encoder y decoder
encoder_inputs = model.input[0]
encoder_lstm = model.get_layer('lstm') 
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = model.input[1]
decoder_lstm = model.get_layer('lstm_1') 
decoder_dense = model.get_layer('dense')  

# Crear los modelos para la predicción (encoder y decoder)
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Interacción con el chatbot
while True:
    input_text = input("Tú: ")
    if input_text.lower() == 'salir':
        print("¡Adiós!")
        break

    # Codificar la entrada del usuario
    input_seq = encode_input_text(input_text)

    # Predecir la respuesta
    decoded_sentence = predict_sequence(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index)

    print(f"Chatbot: {decoded_sentence}")
