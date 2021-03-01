import cv2
import numpy as np
from skimage import io, img_as_ubyte
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
import random
import os
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
import random

if os.path.isdir('foto_simulador') == False:
    print('A pasta "foto_simulador" não existe. Criando diretório.')
else:
    print('A pasta "foto_simulador" existe.')
    images_path = os.getcwd() + '\\foto_simulador'

if os.path.isdir('Aumented_Images') == False:
    print('A pasta "Aumented_Images" não existe. Criando diretório.')
    os.mkdir('Aumented_Images')
else:
    print('A pasta "Aumented_Images" existe.')
    augmented_path = os.getcwd() + '\\Aumented_Images'

images = []
for im in os.listdir(images_path):
    images.append(os.path.join(images_path, im))
images_to_generate = 15  # qtd de imagens que vai gerar
i = 1                   # variavel para inteirar no images_to_generate


def rotacao_ahrr(image):

#Função responsável por fazer a rotação anti-horaria da imagem.
#Entrada: Imagem
#Saída: Imagem rotacionada entre 0 a 180° no sentindo anti-horario

    angle = random.randint(0, 180)
    return rotate(image, angle)

def rotacao_hrr(image):

#Função responsável por fazer a rotação horaria da imagem.
#Entrada: Imagem
#Saída: Imagem rotacionada entre 0 a 180° no sentindo horario

    angle = random.randint(0, 180)
    return rotate(image, -angle)

def hrz_vira(image):

#Função responsável por fazer a inversão horizontal da imagem.
#Entrada: Imagem
#Saída: Imagem invertida no sentido horizontal

    return np.fliplr(image)

def vtc_vira(image):

#Função responsável por fazer a inversão vertical da imagem.
#Entrada: Imagem
#Saída: Imagem invertida no sentido vertical

    return np.flipud(image)

def ruidos_img(image):

#Função responsável por inserir ruídos randomincos do tipo sal e pimenta na imagem.
#Entrada: Imagem
#Saída: Imagem com ruidos do tipo sal e pimenta

    return random_noise(image)

def brilhoo(image):

#Função responsável por incrementar brilho a imagem.
#Entrada: Imagem
#Saída: Imagem com brilho

    brilhin = np.ones(image.shape, dtype="uint8") * 70
    aumentabrilho = cv2.add(image, brilhin)
    return aumentabrilho

def blur_img(image):

#Função responsável por aplicar um filtro mediana na imagem.
#Entrada: Imagem
#Saída: Imagem com filtro mediana

    k_size = random.randrange(1,10,2)
    img_blur = cv2.medianBlur(image, k_size)
    return img_blur

def zoom(image):

#Função responsável por aplicar zoom na imagem.
#Entrada: Imagem
#Saída: Imagem com zoom

    zoom_value = random.random()
    hidth, width = image.shape[:2]
    h_taken = int(zoom_value*hidth)
    w_taken = int(zoom_value*width)
    h_start = random.randint(0, hidth-h_taken)
    w_start = random.randint(0, width-w_taken)
    image = image[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    image = cv2.resize(image, (hidth, width), cv2.INTER_CUBIC)
    return image

# Dicionario para ativar as funcoes
transformations = {'Rotacao anti-horaria': rotacao_ahrr,
                   'Rotacao horaria': rotacao_hrr,
                   'Horizontal flip': hrz_vira,
                   'Vertical flip': vtc_vira,
                   'Ruidos': ruidos_img,
                   'Brilho': brilhoo,
                   'Blur Image': blur_img,
                   'Zoom': zoom
                  }

while i <= images_to_generate:
    image = random.choice(images)
    original_image = io.imread(image)
    transformed_image = []
    n = 0       # variável para iterar até o número de transformação
# escolha um número aleatório de transformação para aplicar na imagem
    transformation_count = random.randint(1, len(transformations))
    while n <= transformation_count:
        # Escolha aleatorio do metodo a ser aplicado
        key = random.choice(list(transformations))
        print(key)
        transformed_image = transformations[key](original_image)
        n += 1
    new_image_path = "%s/augmented_image_%s.jpg" % (augmented_path, i)
    # Converta uma imagem para o formato de byte sem sinal, com valores em [0, 255].
    transformed_image = img_as_ubyte(transformed_image)
    # converter a imagem antes de gravar
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    # Salvar a imagem ja convertida
    cv2.imwrite(new_image_path, transformed_image)
    i = i+1