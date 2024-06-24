from google.colab import files
import os

# Crie diretórios
if not os.path.exists('imagens'):
    os.makedirs('imagens')

if not os.path.exists('rostos_detectados'):
    os.makedirs('rostos_detectados')

# Faça upload das imagens
uploaded = files.upload()
for filename in uploaded.keys():
    os.rename(filename, os.path.join('imagens', filename))