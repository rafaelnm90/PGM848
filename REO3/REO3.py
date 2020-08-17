###### CAPA DE APRESENTAÇÃO #####
print('=-'*78)
print('Aluno: Rafael Novais de Miranda',' '*50,'Disciplina: AVANÇOS CIENTÍFICOS EM GENÉTICA E MELHORAMENTO DE PLANTAS I')
print('Professor: Vinicius Quintão Carneiro',' '*45,'Programa: Genética e Melhoramento de Plantas')
print('=-'*78)
print(' '*50, 'REO 03 - LISTA DE EXERCÍCIOS')
print('=-'*78)
print(' ')

##### Pacotes utilizados #####
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd

##### Número do exercício #####
print('EXERCÍCIO 01')
print('Selecione uma imagem a ser utilizada no trabalho prático e realize os seguintes processos utilizando as bibliotecas OPENCV e Scikit-Image do Python:')
print(' ')

##### Questão 1.a #####
print('a) Aplique o filtro de média com cinco diferentes tamanhos de kernel e compare os resultados com a imagem original;')
print('Resposta:')
arquivo = "img.jpg"
img_bgr = cv2.imread(arquivo, 1) #0-->Imagem intensidade, 1-->Imagem original
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
#Filtros - Média
img_fm_1 = cv2.blur(img_rgb, (3,3))
img_fm_2 = cv2.blur(img_rgb, (7,7))
img_fm_3 = cv2.blur(img_rgb, (9,9))
img_fm_4 = cv2.blur(img_rgb, (11,11))
img_fm_5 = cv2.blur(img_rgb, (13,13))
#Apresentação das imagens
plt.figure('Questão 1.a')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('Imagem Original')

plt.subplot(2,3,2)
plt.imshow(img_fm_1)
plt.xticks([])
plt.yticks([])
plt.title('03x03')

plt.subplot(2,3,3)
plt.imshow(img_fm_2)
plt.xticks([])
plt.yticks([])
plt.title('07x07')

plt.subplot(2,3,4)
plt.imshow(img_fm_3)
plt.xticks([])
plt.yticks([])
plt.title('09x09')

plt.subplot(2,3,5)
plt.imshow(img_fm_4)
plt.xticks([])
plt.yticks([])
plt.title('11x11')

plt.subplot(2,3,6)
plt.imshow(img_fm_5)
plt.xticks([])
plt.yticks([])
plt.title('13x13')
plt.show()
print('-'*50)
print(' ')

##### Questão 1.b #####
print('b) Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel e compare os resultados entre si e com a imagem original.')
print('Resposta:')
arquivo = "img.jpg"
img_bgr = cv2.imread(arquivo, 1)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
#Filtros - Média
img_fm_2 = cv2.blur(img_rgb, (7,7))
img_fm_5 = cv2.blur(img_rgb, (13,13))
#Filtros - Gaussiano
img_fg_2 = cv2.GaussianBlur(img_rgb, (7,7), 0) #média ponderada
img_fg_5 = cv2.GaussianBlur(img_rgb, (13,13), 0) #média ponderada
#Filtros - mediana
img_fmed_2 = cv2.medianBlur(img_rgb, 7)
img_fmed_5 = cv2.medianBlur(img_rgb, 13)
#Filtros - bilateral
img_fb_2 = cv2.bilateralFilter(img_rgb, 7, 7, 3) #tamanho do kernel, intensidade da cor e distancia da ponderação do pixel
img_fb_5 = cv2.bilateralFilter(img_rgb, 13, 13, 3) #tamanho do kernel, intensidade da cor e distancia da ponderação do pixel

#Apresentação das imagens
plt.figure('Questão 1.b')
plt.subplot(5,2,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('Imagem Original')

plt.subplot(5,2,3)
plt.imshow(img_fm_2)
plt.xticks([])
plt.yticks([])
plt.title('Média - 07x07')

plt.subplot(5,2,4)
plt.imshow(img_fm_5)
plt.xticks([])
plt.yticks([])
plt.title('Média - 13x13')

plt.subplot(5,2,5)
plt.imshow(img_fg_2)
plt.xticks([])
plt.yticks([])
plt.title('Gaussiano - 07x07')

plt.subplot(5,2,6)
plt.imshow(img_fg_5)
plt.xticks([])
plt.yticks([])
plt.title('Gaussiano - 13x13')

plt.subplot(5,2,7)
plt.imshow(img_fmed_2)
plt.xticks([])
plt.yticks([])
plt.title('Mediana - 07x07')

plt.subplot(5,2,8)
plt.imshow(img_fmed_5)
plt.xticks([])
plt.yticks([])
plt.title('Mediana - 13x13')

plt.subplot(5,2,9)
plt.imshow(img_fb_2)
plt.xticks([])
plt.yticks([])
plt.title('Bilateral - 07x07')

plt.subplot(5,2,10)
plt.imshow(img_fb_5)
plt.xticks([])
plt.yticks([])
plt.title('Bilateral - 13x13')
plt.show()
print('-'*50)
print(' ')

##### Questão 1.c #####
print('c) Realize a segmentação da imagem utilizando o processo de limiarização. Utilizando o reconhecimento de contornos, identifique e salve os objetos de interesse. Além disso, acesse as bibliotecas Opencv e Scikit-Image, verifique as variáveis que podem ser mensuradas e extraia as informações pertinentes (crie e salve uma tabela com estes dados). Apresente todas as imagens obtidas ao longo deste processo.')
print('Resposta:')
# Leitura da imagem
arquivo = "img.jpg"
img_bgr = cv2.imread(arquivo, 1)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

#Separação dos canais
R,G,B = cv2.split(img_rgb)
(L, img_limiar_inv) = cv2.threshold(R,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Imagem segmentada
img_segmentada = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar_inv)
mascara = img_limiar_inv.copy()
cnts,h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
dimen = []

#Informações das imagens
for (i, c) in enumerate(cnts):

	(x, y, w, h) = cv2.boundingRect(c)
	obj = img_limiar_inv[y:y+h,x:x+w]
	img_rgb = img_segmentada[y:y+h,x:x+w]
	obj_rgb = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)
	cv2.imwrite('s'+str(i+1)+'.png',obj_rgb)
	cv2.imwrite('sb'+str(i+1)+'.png',obj)

	regiao = regionprops(obj)
	print('Semente: ', str(i+1))
	print('Dimensão da Imagem: ', np.shape(obj))
	print('Medidas Físicas')
	print('Centroide: ', regiao[0].centroid)
	print('Comprimento do eixo menor: ', regiao[0].minor_axis_length)
	print('Comprimento do eixo maior: ', regiao[0].major_axis_length)
	print('Razão: ', regiao[0].major_axis_length / regiao[0].minor_axis_length)
	area = cv2.contourArea(c)
	print('Área: ', area)
	print('Perímetro: ', cv2.arcLength(c,True))

	print('Medidas de Cor')
	min_val_R, max_val_R, min_loc_R, max_loc_R = cv2.minMaxLoc(img_rgb[:,:,0], mask=obj)
	print('Valor Mínimo no R: ', min_val_R, ' - Posição: ', min_loc_R)
	print('Valor Máximo no R: ', max_val_R, ' - Posição: ', max_loc_R)
	med_val_R = cv2.mean(img_rgb[:,:,0], mask=obj)
	print('Média no Componente R: ', med_val_R)

	min_val_G, max_val_G, min_loc_G, max_loc_G = cv2.minMaxLoc(img_rgb[:, :, 1], mask=obj)
	print('Valor Mínimo no G: ', min_val_G, ' - Posição: ', min_loc_G)
	print('Valor Máximo no G: ', max_val_G, ' - Posição: ', max_loc_G)
	med_val_G = cv2.mean(img_rgb[:,:,1], mask=obj)
	print('Média no Componente G: ', med_val_G)

	min_val_B, max_val_B, min_loc_B, max_loc_B = cv2.minMaxLoc(img_rgb[:, :, 2], mask=obj)
	print('Valor Mínimo no Cb: ', min_val_B, ' - Posição: ', min_loc_B)
	print('Valor Máximo no Cb: ', max_val_B, ' - Posição: ', max_loc_B)
	med_val_B = cv2.mean(img_rgb[:,:,2], mask=obj)
	print('Média no Croma com diferença azul: ', med_val_B)
	print('-'*50)

razao = regiao[0].major_axis_length / regiao[0].minor_axis_length
dimen += [[str(i + 1), str(h), str(w), str(area), str(razao)]]
dados = pd.DataFrame(dimen)
dados = dados.rename(columns={0: 'Grãos', 1: 'Altura', 2: 'Largura', 3: 'Area', 4: 'Razão'})
dados.to_csv('Medidas.csv', index=False)

print('Total de sementes: ', len(cnts))
print('-'*50)

seg = img_segmentada.copy()
cv2.drawContours(seg,cnts,-1,(0,255,0),2)

plt.figure('Questão 1.c')
plt.subplot(1,2,1)
plt.imshow(seg)
plt.xticks([])
plt.yticks([])
plt.title('Contornos')

plt.subplot(1,2,2)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('Arroz')
plt.show()
print('-'*50)
print(' ')

##### Questão 1.d #####
print('d) Utilizando máscaras, apresente o histograma somente dos objetos de interesse.')
print('Resposta:')

s82 = 's82.png'
s82_bgr = cv2.imread(s82, 1)
s82 = cv2.cvtColor(s82_bgr, cv2.COLOR_BGR2RGB)
sb82 = "sb82.png"
sb82_bgr = cv2.imread(sb82, 0)
hist_segmentada_R_1 = cv2.calcHist([s82],[0], sb82_bgr, [256],[0,256])
hist_segmentada_G_1 = cv2.calcHist([s82],[1], sb82_bgr, [256],[0,256])
hist_segmentada_B_1 = cv2.calcHist([s82],[2], sb82_bgr, [256],[0,256])

s32 = 's32.png'
s32_bgr = cv2.imread(s32, 1)
s32 = cv2.cvtColor(s32_bgr, cv2.COLOR_BGR2RGB)
sb32 = "sb32.png"
sb32_bgr = cv2.imread(sb32, 0)
hist_segmentada_R_2 = cv2.calcHist([s32],[0], sb32_bgr, [256],[0,256])
hist_segmentada_G_2 = cv2.calcHist([s32],[1], sb32_bgr, [256],[0,256])
hist_segmentada_B_2 = cv2.calcHist([s32],[2], sb32_bgr, [256],[0,256])

plt.figure('Questão 1.d-1/2')
plt.subplot(3,3,2)
plt.imshow(s82)
plt.title('Objeto: 1 - s82')

plt.subplot(3, 3, 4)
plt.imshow(s82[:,:,0],cmap='gray')
plt.title('Objeto: 1 - R')

plt.subplot(3, 3, 5)
plt.imshow(s82[:,:,1],cmap='gray')
plt.title('Objeto: 1 - G')

plt.subplot(3, 3, 6)
plt.imshow(s82[:,:,2],cmap='gray')
plt.title('Objeto: 1 - B')

plt.subplot(3, 3, 7)
plt.plot(hist_segmentada_R_1, color='r')
plt.title("Histograma - R")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 8)
plt.plot(hist_segmentada_G_1, color='g')
plt.title("Histograma - G")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 9)
plt.plot(hist_segmentada_B_1, color='b')
plt.title("Histograma - B")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

plt.figure('Questão 1.d-2/2')
plt.subplot(3,3,2)
plt.imshow(s32)
plt.title('Objeto: 2 - s32')

plt.subplot(3, 3, 4)
plt.imshow(s32[:,:,0],cmap='gray')
plt.title('Objeto: 2 - R')

plt.subplot(3, 3, 5)
plt.imshow(s32[:,:,1],cmap='gray')
plt.title('Objeto: 2 - G')

plt.subplot(3, 3, 6)
plt.imshow(s32[:,:,2],cmap='gray')
plt.title('Objeto: 2 - B')

plt.subplot(3, 3, 7)
plt.plot(hist_segmentada_R_2, color='r')
plt.title("Histograma - R")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 8)
plt.plot(hist_segmentada_G_2, color='g')
plt.title("Histograma - G")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 9)
plt.plot(hist_segmentada_B_2, color='b')
plt.title("Histograma - B")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
print('-'*50)
print(' ')

##### Questão 1.e #####
print('e) Realize a segmentação da imagem utilizando a técnica de k-means. Apresente as imagens obtidas neste processo.')
print('Resposta:')
print('INFORMAÇÕES')
print('-'*80)
arquivo = "img.jpg"
img_bgr = cv2.imread(arquivo, 1)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
print('Dimensão: ',np.shape(img_rgb))
print(np.shape(img_rgb)[0], ' x ',np.shape(img_rgb)[1], ' = ', np.shape(img_rgb)[0] * np.shape(img_rgb)[1])
print('-'*80)

pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print('-'*80)
print('Dimensão Matriz: ',pixel_values.shape)
print('-'*80)

# K-means
# Critério de Parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
# Número de Grupos (k)
k = 2
dist, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print('-'*80)
print('SQ das Distâncias de Cada Ponto ao Centro: ', dist)
print('-'*80)
print('Dimensão labels: ', labels.shape)
print('Valores únicos: ',np.unique(labels))
print('Tipo labels: ', type(labels))

# flatten the labels array
labels = labels.flatten()
print('-'*80)
print('Dimensão flatten labels: ', labels.shape)
print('Tipo labels (f): ', type(labels))
print('-'*80)

# Valores dos labels
val_unicos,contagens = np.unique(labels,return_counts=True)
val_unicos = np.reshape(val_unicos,(len(val_unicos),1))
contagens = np.reshape(contagens,(len(contagens),1))
hist = np.concatenate((val_unicos,contagens),axis=1)
print('Histograma')
print(hist)
print('-'*80)
print('Centroides Decimais')
print(centers)
print('-'*80)

# Conversão dos centroides para valores de interos de 8 digitos
centers = np.uint8(centers)
print('-'*80)
print('Centroides uint8')
print(centers)
print('-'*80)

# Conversão dos pixels para a cor dos centroides
matriz_segmentada = centers[labels]
print('-'*80)
print('Dimensão Matriz Segmentada: ',matriz_segmentada.shape)
print('Matriz Segmentada')
print(matriz_segmentada[0:5,:])
print('-'*80)

# Reformatar a matriz na imagem de formato original
img_segmentada = matriz_segmentada.reshape(img_rgb.shape)

# Grupo 1
original_01 = np.copy(img_rgb)
matriz_or_01 = original_01.reshape((-1, 3))
matriz_or_01[labels != 1] = [0, 0, 0]
img_final_01 = matriz_or_01.reshape(img_rgb.shape)

# Grupo 2
original_02 = np.copy(img_rgb)
matriz_or_02 = original_02.reshape((-1, 3))
matriz_or_02[labels == 1] = [0, 0, 0]
img_final_02 = matriz_or_02.reshape(img_rgb.shape)
########################################################################################################################

# Apresentar Imagem
plt.figure('Questão 1.e')
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title('Imagem RGB')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_segmentada)
plt.title('Rótulos')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_final_01)
plt.title('Grupo 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_final_02)
plt.title('Grupo 2')
plt.xticks([])
plt.yticks([])
plt.show()
print('-'*50)
print(' ')

##### Questão 1.f #####
print('f) Realize a segmentação da imagem utilizando a técnica de watershed. Apresente as imagens obtidas neste processo.')
print('Resposta:')

r,g,b = cv2.split(img_rgb)
limiar_1f, mask = cv2.threshold(r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_dist = ndimage.distance_transform_edt(mask)
localmax = peak_local_max(img_dist, indices=False, min_distance=5, labels=mask)

print('Quantidade de Picos')
print(np.unique(localmax,return_counts=True))
print('-'*50)

marcadores,n_marcadores = ndimage.label(localmax, structure=np.ones((3, 3)))

print('Marcadores')
print(np.unique(marcadores,return_counts=True))
print('-'*50)
img_ws = watershed(-img_dist, marcadores, mask=mask)

print('Watershed')
print(np.unique(img_ws,return_counts=True))
print("Número de sementes: ", len(np.unique(img_ws)) - 1)
img_final = np.copy(img_rgb)
img_final[img_ws != 38] = [0,0,0]

plt.figure('Questão 1.f')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('RGB')

plt.subplot(2,3,2)
plt.imshow(r,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('R')

plt.subplot(2,3,3)
plt.imshow(mask,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Máscara binária')

plt.subplot(2,3,4)
plt.imshow(img_dist,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Imagem distância')

plt.subplot(2,3,5)
plt.imshow(img_ws,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Imagem folha segmentada')

plt.subplot(2,3,6)
plt.imshow(img_final)
plt.xticks([])
plt.yticks([])
plt.title('Seleção')
plt.show()
print('-'*50)
print(' ')

##### Questão 1.g #####
print('g) Compare os resultados das três formas de segmentação (limiarização, k-means e watershed) e identifique as potencialidades de cada delas.')
print('Resposta:')
plt.figure('Questão 1.g')
plt.subplot(1,3,1)
plt.imshow(mask,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('OTSU')

plt.subplot(1,3,2)
plt.imshow(img_final_01)
plt.title('K-MEANS')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(img_ws)
plt.xticks([])
plt.yticks([])
plt.title('Wathershed')
plt.show()
print('-'*50)
print(' ')
