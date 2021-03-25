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
import csv
import xml.etree.ElementTree as ET


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
images_to_generate = images_to_generate*2
i = 0                   # variavel para inteirar no images_to_generate


def minmax(img2):
    # Create a black image
    #img = np.zeros((200,300,3), np.uint8)
    #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,255,255),-1)
 
    #MUDANCAS

    # Find Canny edges 
    edged = cv2.Canny(img2, 200, 200) 
    #cv2.imshow('img2', edged)
    #cv2.waitKey(0) 
  
    # Finding Contours 
    contours, hierarchy =  cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
  
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        #print("find")
        #print(str(x) +'-'+str(y) +'-'+str(w+x) +'-'+str(y+h) )

    #print("Number of Contours found = " + str((contours))) 
    cf = contours
    # Draw all contours 
    # -1 signifies drawing all contours 
    #cv2.drawContours(img2, contours, -1, (0, 255, 0), 3) 
    if (cf == []):
        x = 0
        y = 0
        w = 0
        h = 0

    #cv2.imshow('img2', img2)

    #cv2.waitKey(0) 
    return(x, y, w+x, y+h)


def rotacao(image):
    # points for test.jpg
    cnt = np.array([
            [[750, 250]],
            [[0, 500]],
            [[1000, 1050]],
            [[1000, 500]]
        ])
    
    rect = cv2.minAreaRect(cnt)
    

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped
   

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

    #return random_noise(image)
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.05
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
    out[coords] = 0
    #cv2.imshow('blabs', out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return out

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
transformations = {'Rotacao': rotacao,
                   'Horizontal flip': hrz_vira,
                   'Vertical flip': vtc_vira,
                   'Ruidos': ruidos_img,
                   'Brilho': brilhoo,
                   'Blur Image': blur_img,
                   'Zoom': zoom
                  }



data = []
while i < images_to_generate:
    #colocar o numero de imagens + arquivos que tem, para que as imagens possam ser escolhidas aleatoriamente
    x = random.randrange(147)

    arquivo = ET.parse(images[i])
    if x == 0:
        arquivo = ET.parse(images[x])
    elif x % 2 == 0:
        arquivo = ET.parse(images[x])
    else:
        x = x + 1
        arquivo = ET.parse(images[x])

    bla = arquivo.getroot()

    numeros = bla.findall("object/bndbox")
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0

    for item in numeros:
        xmin = int(item.find("xmin").text)
        ymin = int(item.find("ymin").text)
        xmax = int(item.find("xmax").text)
        ymax = int(item.find("ymax").text)
    

    print(xmin, ymin, xmax, ymax)


    image = images[x + 1]
    original_image = io.imread(image)
    transformed_image = []
    n = 0       # variável para iterar até o número de transformação

    height, width, channels = original_image.shape
    # imagemm - cria uma img preta com um bounding no lugar q deveria estar na imagem da pista
    imagemm = np.zeros((height,width,channels), np.uint8)
    cv2.rectangle(imagemm,(xmin,ymin),(xmax,ymax),(255,255,255),-1)

    #cv2.imshow('img1', imagemm)
    #cv2.waitKey(0)
# escolha um número aleatório de transformação para aplicar na imagem
    transformation_count = random.randint(1, len(transformations))
    
    while n <= transformation_count:
        # Escolha aleatorio do metodo a ser aplicado
        key = random.choice(list(transformations))
        print(key)
        transformed_image = transformations[key](original_image)
        #print(transformed_image.dtype)
        # faz as mesmas transformacoes na imagem preta
        img2 = transformations[key](imagemm)

        # mean normalization
        image = transformed_image.astype(np.float32) / 255
        image -= image.mean()
        image /= image.std()
        transformed_image = transformed_image.astype(np.uint8)
        transformed_image = np.round(transformed_image).astype(np.uint8)

        n += 1
    

    #cv2.imshow('blabs', transformed_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

    nome = "augmented_image_%s.jpg" % (i)
    new_image_path = "%s/augmented_image_%s.jpg" % (augmented_path, i)
    # Converta uma imagem para o formato de byte sem sinal, com valores em [0, 255].
    transformed_image = img_as_ubyte(transformed_image)
    img2 = img_as_ubyte(img2)
    #cv2.imshow('blabs', transformed_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # converter a imagem antes de gravar
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    #cv2.imshow('blabs', transformed_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    # Salvar a imagem ja convertida
    cv2.imwrite(new_image_path, transformed_image)
    i = i+2
    height, width, channels = transformed_image.shape

    # chama a funcao que vai retornar o x & y min e max do bounding
    xmin, ymin, xmax, ymax = minmax(img2)
        
    # elemento da lista data[] que armazena as informacoes da imagem transformada
    tentativa = (nome, height, width, 'pista', xmin, ymin, xmax, ymax)
    
    #acrescenta "tentativa" a lista data[] (cada imagem gera uma tentativa diferente q e acrescido a data)
    data.append(tentativa)

    #cv2.imshow('img2', img2)
    #cv2.waitKey(0) 

# cria o documento csv que armazena os dados por imagem
with open('teste.csv', "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
        writer.writerows(data)    
