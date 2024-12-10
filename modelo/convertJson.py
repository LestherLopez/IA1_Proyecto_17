import pickle
import json

# Cargar palabras y clases
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Guardar como JSON
with open('words.json', 'w') as f:
    json.dump(words, f)

with open('classes.json', 'w') as f:
    json.dump(classes, f)

print("Conversion completada: words.json y classes.json creados.")
