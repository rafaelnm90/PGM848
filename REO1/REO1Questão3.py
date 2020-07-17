##### Pacotes utilizados #####
import numpy as np

##### Questão 3.a #####
def media (vetor):
    somador = 0
    contagem_de_dados = 0
    it = 0
    for i in vetor:
        somador = somador + i
        it = it + 1
        contagem_de_dados = contagem_de_dados + 1
    mean = somador/contagem_de_dados
    return mean

def var_amostral (vetor):
    somador = 0
    it = 0
    sdq = 0
    ne = 0
    for i in vetor:
        somador = somador + i
        it = it + 1
        ne = ne + 1
        sdq = sdq + i**2
    var = (sdq - ((somador**2/ne)))/(ne-1)
    return var
print(' ')
print('TESTE DA FUNÇÃO: ')
vetor = np.array ([10, 20, 30])
print('O vetor possui os seguintes elementos {}, a sua média é {} e a variância amostral é {}'.format(vetor, media(vetor), var_amostral(vetor)))



