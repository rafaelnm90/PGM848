###### CAPA DE APRESENTAÇÃO #####
print('=-'*78)
print('Aluno: Rafael Novais de Miranda',' '*50,'Disciplina: AVANÇOS CIENTÍFICOS EM GENÉTICA E MELHORAMENTO DE PLANTAS I')
print('Professor: Vinicius Quintão Carneiro',' '*45,'Programa: Genética e Melhoramento de Plantas')
print('=-'*78)
print(' '*50, 'REO 02 - LISTA DE EXERCÍCIOS')
print('=-'*78)
print(' ')

##### Pacotes utilizados #####
import cv2
import numpy as np
from matplotlib import pyplot as plt

##### Número do exercício #####
print('EXERCÍCIO 01')
print('Selecione uma imagem a ser utilizada no trabalho prático e realize os seguintes processos utilizando o pacote OPENCV do Python:')
print(' ')

##### Questão 1.a #####
print('a) Apresente a imagem e as informações de número de linhas e colunas; número de canais e número total de pixels;')
print('Resposta:')
arquivo = "img.jpg"
img_bgr = cv2.imread(arquivo, 1) #0-->Imagem intensidade, 1-->Imagem original
img_org = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
plt.figure('Imagem 1.a') #nome do arquivo que irá gerar
plt.imshow(img_org) #cmap --> escala de cor da intensidade
plt.xticks([])
plt.yticks([])
plt.title('Imagem 1.a') #Título que apresentará dentro da imagem
plt.show()
print('     INFORMAÇÕES')
lin, col, canais = np.shape(img_org)
print('Tipo de imagem: {}'.format(img_org.dtype))
print('Dimensão da imagem: {}x{}'.format(lin, col))
print('Largura: {}'.format(col))
print('Altura: {}'.format(lin))
print('Canais: {}'.format(canais))
print(('Pixels: {}'.format((lin*col))))
print('-'*50)
print(' ')

##### Questão 1.b #####
print('b) Faça um recorte da imagem para obter somente a área de interesse. Utilize esta imagem para a solução das próximas alternativas;')
print('Resposta:')
img_rec = img_org[109:370, 160:433]

plt.figure('Imagem 1.b') #nome do arquivo que irá gerar
plt.subplot(1,2,1)
plt.imshow(img_rec)
plt.title('Imagem Recortada') #Título que apresentará dentro da imagem

plt.subplot(1,2,2)
plt.imshow(img_org)
plt.title('Imagem Original') #Título que apresentará dentro da imagem
plt.show()
print('A resposta foi a Imagem 1.b gerada')
print('-'*50)
print(' ')

##### Questão 1.c #####
print('c) Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando os mapas de cores “Escala de Cinza” e “JET”')
print('Resposta:')
img_gray = cv2.cvtColor(img_rec,cv2.COLOR_RGB2GRAY)

plt.figure('Imagem 1.c')
plt.subplot(1,3,1)
plt.imshow(img_org)
plt.title('Imagem Original')

plt.subplot(1,3,2)
plt.imshow(img_gray, cmap="gray")
plt.colorbar(orientation='horizontal')
plt.title('Imagem Escala de Cinza')

plt.subplot(1,3,3)
plt.imshow(img_gray, cmap="jet")
plt.colorbar(orientation='horizontal')
plt.title('Imagem JET')
plt.show()
print('-'*50)
print(' ')

##### Questão 1.d #####
print('d) Apresente a imagem em escala de cinza e o seu respectivo histograma; Relacione o histograma e a imagem.')
print('Resposta:')
hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.figure('Imagem 1.d') #nome do arquivo que irá gerar
plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray") #cmap --> escala de cor da intensidade
plt.colorbar(orientation='horizontal')
plt.title('Escala de Cinza') #Título que apresentará dentro da imagem

plt.subplot(1,2,2)
plt.plot(hist, color="black")
plt.title('Histograma') #Título que apresentará dentro da imagem
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')
plt.show()
print('Histograma')
print(hist)
print(' ')
dim = len(hist)
print('Dimensão do histograma: {}'.format(dim))
print('Portanto por ser uma imagem não binária, ela possui níveis diferentes de cinza, com diferentes intensidades  entre os níveis de preto e branco. Mas dentro desse histograma observamos que, posssuímos uma grande intensidade de pixels na faixa de 250-255, e também, uma baixa porém, a única restante, na faixa de 25-125')
print('-'*50)
print(' ')

##### Questão 1.e #####
print('e) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de modo a remover o fundo da imagem utilizando um limiar manual e o limiar obtido pela técnica de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação. Explique os resultados.')
print('Resposta:')
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
limiar_cinza_gray = 140

(L, img_limiar_gray) = cv2.threshold(img_gray, limiar_cinza_gray, 255, cv2.THRESH_BINARY)
(L_gray_otsu, img_otsu_gray) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img_segment_gray = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar_gray)
img_gray_otsu = cv2.bitwise_and(img_rec,img_rec,mask=img_otsu_gray)

hist_gray = cv2.calcHist([img_segment_gray],[0],img_limiar_gray, [256], [0,256])

plt.figure('Imagem 1.e') #nome do arquivo que irá gerar
plt.subplot(4,2,1)
plt.imshow(img_gray, cmap="gray") #cmap --> escala de cor da intensidade
plt.xticks([])
plt.yticks([])
plt.title('Escala de Cinza') #Título que apresentará dentro da imagem

plt.subplot(4,2,3)
plt.plot(hist_gray, color="black")
plt.axvline(x=limiar_cinza_gray, color= 'black')
plt.xlim([0, 256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.title('Histograma - Cinza') #Título que apresentará dentro da imagem

plt.subplot(4,2,5)
plt.imshow(img_limiar_gray, cmap='gray')
plt.title('Binário Manual - L: {}'.format(limiar_cinza_gray))

plt.subplot(4,2,2)
plt.imshow(img_gray, cmap="gray") #cmap --> escala de cor da intensidade
plt.xticks([])
plt.yticks([])
plt.title('Escala de Cinza') #Título que apresentará dentro da imagem

plt.subplot(4,2,4)
plt.plot(hist_gray, color="black")
plt.axvline(x=L_gray_otsu, color= 'black')
plt.xlim([0, 256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.title('Histograma - Cinza') #Título que apresentará dentro da imagem

plt.subplot(4,2,6)
plt.imshow(img_otsu_gray, cmap='gray')
plt.title('Binário OTSU - L: {}'.format(L_gray_otsu))

plt.subplot(4,2,7)
plt.imshow(img_segment_gray)
plt.title("Imagem Segmentada Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(4,2,8)
plt.imshow(img_gray_otsu)
plt.title("Imagem Segmentada OTSU")
plt.xticks([])
plt.yticks([])
plt.show()
print('-'*50)
print(' ')

##### Questão 1.f #####
print('f) Apresente uma figura contento a imagem selecionada nos sistemas RGB, Lab, HSV e YCrCb.')
print('Resposta:')
img_lab = cv2.cvtColor(img_rec, cv2.COLOR_RGB2LAB)
img_hsv = cv2.cvtColor(img_rec, cv2.COLOR_RGB2HSV)
img_ycrcb = cv2.cvtColor(img_rec, cv2.COLOR_RGB2YCR_CB)
plt.figure('Imagem 1.f')
plt.subplot(2,2,1)
plt.imshow(img_rec)
plt.title('Imagem RGB')

plt.subplot(2,2,2)
plt.imshow(img_lab)
plt.title('Imagem Lab')

plt.subplot(2,2,3)
plt.imshow(img_hsv)
plt.title('Imagem HSV')

plt.subplot(2,2,4)
plt.imshow(img_ycrcb)
plt.title('Imagem YCrCb')
plt.show()
print('-'*50)
print(' ')

##### Questão 1.g #####
print('g) Apresente uma figura para cada um dos sistemas de cores (RGB, HSV, Lab e YCrCb) contendo a imagem de cada um dos canais e seus respectivos histogramas.')
print('Resposta:')

#Sistema RGB
hist_r = cv2.calcHist([img_rec], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img_rec], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([img_rec], [2], None, [256], [0, 256])

plt.figure('Questão 1.g - RGB (1/4)')
plt.subplot(3, 3, 2)
plt.imshow(img_rec, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title('Imagem RGB')

plt.subplot(3, 3, 4)
plt.imshow(img_rec[:, :, 0], cmap='gray')
plt.title('R')

plt.subplot(3, 3, 7)
plt.plot(hist_r, color="r")
plt.title('Histograma - R')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 5)
plt.imshow(img_rec[:, :, 1], cmap='gray')
plt.title('G')

plt.subplot(3, 3, 8)
plt.plot(hist_g, color="g")
plt.title('Histograma - G')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 6)
plt.imshow(img_rec[:, :, 2], cmap='gray')
plt.title('B')

plt.subplot(3, 3, 9)
plt.plot(hist_b, color="b")
plt.title('Histograma - B')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.show()

#Sistema HSV
hist_h = cv2.calcHist([img_hsv], [0], None, [256], [0, 256])
hist_s = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
hist_v = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])

plt.figure('Questão 1.g - HSV (2/4)')
plt.subplot(3, 3, 2)
plt.imshow(img_hsv, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title('Imagem HSV')

plt.subplot(3, 3, 4)
plt.imshow(img_hsv[:, :, 0], cmap='gray')
plt.title('H')

plt.subplot(3, 3, 7)
plt.plot(hist_h, color="black")
plt.title('Histograma - H')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 5)
plt.imshow(img_hsv[:, :, 1], cmap='gray')
plt.title('S')

plt.subplot(3, 3, 8)
plt.plot(hist_s, color="black")
plt.title('Histograma - S')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 6)
plt.imshow(img_hsv[:, :, 2], cmap='gray')
plt.title('V')

plt.subplot(3, 3, 9)
plt.plot(hist_v, color="black")
plt.title('Histograma - V')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.show()

#Sistema LAB
hist_l = cv2.calcHist([img_lab], [0], None, [256], [0, 256])
hist_a = cv2.calcHist([img_lab], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([img_lab], [2], None, [256], [0, 256])

plt.figure('Questão 1.g - LAB (3/4)')
plt.subplot(3, 3, 2)
plt.imshow(img_lab, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title('Imagem LAB')

plt.subplot(3, 3, 4)
plt.imshow(img_lab[:, :, 0], cmap='gray')
plt.title('L')

plt.subplot(3, 3, 7)
plt.plot(hist_l, color="black")
plt.title('Histograma - L')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 5)
plt.imshow(img_lab[:, :, 1], cmap='gray')
plt.title('A')

plt.subplot(3, 3, 8)
plt.plot(hist_a, color="black")
plt.title('Histograma - A')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 6)
plt.imshow(img_lab[:, :, 2], cmap='gray')
plt.title('B')

plt.subplot(3, 3, 9)
plt.plot(hist_b, color="black")
plt.title('Histograma - B')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.show()

#Sistema YCrCb
hist_Y = cv2.calcHist([img_ycrcb], [0], None, [256], [0, 256])
hist_Cr = cv2.calcHist([img_ycrcb], [1], None, [256], [0, 256])
hist_Cb = cv2.calcHist([img_ycrcb], [2], None, [256], [0, 256])

plt.figure('Questão 1.g - YCrCb (4/4)')
plt.subplot(3, 3, 2)
plt.imshow(img_ycrcb, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title('Imagem YCrCb')

plt.subplot(3, 3, 4)
plt.imshow(img_ycrcb[:, :, 0], cmap='gray')
plt.title('Y')

plt.subplot(3, 3, 7)
plt.plot(hist_Y, color="black")
plt.title('Histograma - Y')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 5)
plt.imshow(img_ycrcb[:, :, 1], cmap='gray')
plt.title('Cr')

plt.subplot(3, 3, 8)
plt.plot(hist_Cr, color="black")
plt.title('Histograma - Cr')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')

plt.subplot(3, 3, 6)
plt.imshow(img_ycrcb[:, :, 2], cmap='gray')
plt.title('Cb')

plt.subplot(3, 3, 9)
plt.plot(hist_Cb, color="black")
plt.title('Histograma - Cb')
plt.xlim([0,256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.show()
print('-'*50)
print(' ')

##### Questão 1.h #####
print('h) Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagem de modo a remover o fundo da imagem utilizando limiar manual e limiar obtido pela técnica de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação. Explique os resultados e sua escolha pelo sistema de cor e canal utilizado na segmentação. Nesta questão apresente a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação.')
print('Resposta:')
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
limiar_cinza = 140
(L, img_limiar) = cv2.threshold(img_gray, limiar_cinza, 255, cv2.THRESH_BINARY)

r, g, b = cv2.split(img_rec)
hist_r = cv2.calcHist([img_rec], [0], None, [256], [0, 256])
limiar_r = 138
(L_r, img_limiar_r)= cv2.threshold(r, limiar_r, 255, cv2.THRESH_BINARY)
hist_g = cv2.calcHist([img_rec], [1], None, [256], [0, 256])
limiar_g = 130
(L_g, img_limiar_g)= cv2.threshold(g, limiar_g, 255, cv2.THRESH_BINARY)
hist_b = cv2.calcHist([img_rec], [2], None, [256], [0, 256])
limiar_b = 145
(L_b, img_limiar_b)= cv2.threshold(b, limiar_b, 255, cv2.THRESH_BINARY)

img_segment_gray = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar)
img_segment_r = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar_r)
img_segment_g = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar_g)
img_segment_b = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar_b)

(L_gray_otsu, img_otsu_gray) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_r, img_limiar_r)= cv2.threshold(r, limiar_r, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_g, img_limiar_g)= cv2.threshold(g, limiar_g, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_b, img_limiar_b)= cv2.threshold(b, limiar_b, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img_segment_gray = cv2.bitwise_and(img_rec,img_rec,mask=img_otsu_gray)
img_segment_r = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar_r)
img_segment_g = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar_g)
img_segment_b = cv2.bitwise_and(img_rec,img_rec,mask=img_limiar_b)

plt.figure('Imagem 1.h - R - Limiarização Manual e OTSU') #nome do arquivo que irá gerar
plt.subplot(4,2,1)
plt.imshow(r, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('R')

plt.subplot(4,2,3)
plt.plot(hist_r, color="r")
plt.axvline(x=limiar_r, color= 'r')
plt.xlim([0, 256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.title('Histograma - R')

plt.subplot(4,2,5)
plt.imshow(img_limiar_r, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title('Binário - L: {}'.format(limiar_r))

plt.subplot(4,2,7)
plt.imshow(img_segment_r)
plt.title("Imagem Segmentada Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(4,2,2)
plt.imshow(r, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('R')

plt.subplot(4,2,4)
plt.plot(hist_r, color="r")
plt.axvline(x=L_r, color= 'r')
plt.xlim([0, 256])
plt.xlabel('Valores de Pixel')
plt.ylabel('Número de Pixel')
plt.title('Histograma - R')

plt.subplot(4,2,6)
plt.imshow(img_limiar_r, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title('OTSU - L: {}'.format(L_r))

plt.subplot(4,2,8)
plt.imshow(img_segment_r)
plt.title("Imagem Segmentada OTSU")
plt.xticks([])
plt.yticks([])
plt.show()
print('-'*50)
print(' ')


##### Questão 1.i #####
print('i) Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara a imagem limiarizada (binarizada) da letra h.')
print('Resposta:')

hist_r = cv2.calcHist([img_rec],[0],img_limiar_r,[256],[0,256])
hist_g = cv2.calcHist([img_rec],[1],img_limiar_r,[256],[0,256])
hist_b = cv2.calcHist([img_rec],[2],img_limiar_r,[256],[0,256])

plt.figure('Imagem 1.i')
plt.subplot(2,3,1)
plt.imshow(img_rec[:,:,0],cmap = 'gray')
plt.title('Segmentada - R')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_rec[:,:,1],cmap = 'gray')
plt.title('Segmentada - G')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_rec[:,:,2],cmap = 'gray')
plt.title('Segmentada - B')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_r,color = 'r')
plt.title("Histograma - R")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,5)
plt.plot(hist_g,color = 'black')
plt.title("Histograma - G")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,6)
plt.plot(hist_b,color = 'b')
plt.title("Histograma - B")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
print('-'*50)
print(' ')

##### Questão 1.j #####
print('j) Realize operações aritméticas na imagem em RGB de modo a realçar os aspectos de seu interesse. Exemplo (2*R-0.5*G). Explique a sua escolha pelas operações aritméticas. Segue abaixo algumas sugestões')
print('Resposta:')
imgOPR1 = 1.7* img_org[:, :, 0] - 1.2* img_org[:, :, 1]
imgOPR = imgOPR1.astype(np.uint8) #convertendo para inteiro de 8bit
histw = cv2.calcHist([imgOPR], [0], None, [256], [0, 256])
(M, img_OTSU) = cv2.threshold(imgOPR, 0, 256, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
img_SEGM = cv2.bitwise_and(img_org, img_org, mask=img_OTSU)

# Apresentando a imagem
plt.figure('Imagem 1.j')
plt.subplot(2, 3, 1)
plt.imshow(img_org, cmap='gray')
plt.title('RGB')

plt.subplot(2, 3, 2)
plt.imshow(imgOPR, cmap='gray')
plt.title('1,7R - 1,2*G')

plt.subplot(2, 3, 3)
plt.plot(histw, color='black')
# plt.axline(x=M, color='black')
plt.xlim([0, 256])
plt.xlabel('Valores de pixels')
plt.xlabel('Número de pixels')

plt.subplot(2, 3, 4)
plt.imshow(img_OTSU, cmap='gray')
plt.title('Imagem binária')

plt.subplot(2, 3, 5)
plt.imshow(img_SEGM, cmap='gray')
plt.title('Imagem segmentada com mascara')
plt.xticks([])
plt.yticks([])
plt.show()
print('-'*50)
print(' ')
