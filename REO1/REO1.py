###### CAPA DE APRESENTAÇÃO #####
print('=-'*78)
print('Aluno: Rafael Novais de Miranda',' '*50,'Disciplina: AVANÇOS CIENTÍFICOS EM GENÉTICA E MELHORAMENTO DE PLANTAS I')
print('Professor: Vinicius Quintão Carneiro',' '*45,'Programa: Genética e Melhoramento de Plantas')
print('=-'*78)
print(' '*50, 'REO 01 - LISTA DE EXERCÍCIOS')
print('=-'*78)
print(' ')

##### Pacotes utilizados #####
import numpy as np
from matplotlib import pyplot as plt

##### Número do exercício #####
print('EXERCÍCIO 01')

##### Questão 1.a #####
print('a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.')
print('Resposta:')
lista = (43.5, 150.30, 17, 28, 35, 79, 20, 99.07, 15)
vetor = np.array(lista)
print('Lista: {}'.format(lista))
print('Vetor: {}'.format(vetor))
print('O vetor é do tipo: {}'.format(type(vetor)))
print('-'*50)
print(' ')

##### Questão 1.b #####
print('b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.')
print('Resposta:')
print('O vetor contém {} elementos.'.format(len(vetor)))
print('A média do vetor é {}'.format(np.mean(vetor)))
print('O valor máximo do vetor é {}'.format(max(vetor)))
print('O valor mínimo do vetor é {}'.format(min(vetor)))
print('A variância deste vetor é {}'.format(np.var(vetor)))
print('-'*50)
print(' ')

##### Questão 1.c #####
print('c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declarado na letra a e o valor da média deste.')
print('Resposta:')
media_vetor = np.mean(vetor)
novo_vetor = np.array((vetor-media_vetor)**2)
tipo_vetor = type(novo_vetor)
print(novo_vetor)
print(' ')
print('O vetor é do tipo: {}'.format(tipo_vetor))
print('-'*50)
print(' ')

##### Questão 1.d #####
print('d) Obtenha um novo vetor que contenha todos os valores superiores a 30.')
print('Resposta:')
print('O vetor que contém apenas valores superiores a 30: {}'.format(vetor[vetor>30]))
print('-'*50)
print(' ')

##### Questão 1.e #####
print('e) Identifique quais as posições do vetor original possuem valores superiores a 30')
print('Resposta:')
print('Vetor boleano indicando posições superiores a 30: {}'.format(vetor>30))
vetor_sup_30 = np.where(vetor>30)
print('As posições do vetor original que possuem valores superiores a 30 são: {}'.format(vetor_sup_30[0]))
print('-'*50)
print(' ')

##### Questão 1.f #####
print('f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.')
print('Resposta:')
prim = vetor [0]
quin = vetor [4]
ult = vetor [-1]
vetor_qf = [prim, quin, ult]
print('O vetor contendo os valores da primeira, quinta e última posição: {}'.format(vetor_qf))
print('-'*50)
print(' ')

##### Questão 1.g #####
print('g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição durante as iterações')
print('Resposta:')
it = 0
print('Considerando que o Python pondera a primeira posição sendo 0:')
for pos,valor in enumerate(vetor):
    it = it+1
    print('Iteração {} --> Na posição {} ocorre o valor {}'.format(it, pos, valor))
print('-'*50)
print(' ')

##### Questão 1.h #####
print('h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.')
print('Resposta:')
def somatorio_quadrado (vetor):
    somador = 0
    it = 0
    for el in vetor:
        print('Elemento: {}'.format(el))
        print('Somador: {}'.format(somador))
        somador = somador + (el)**2
        it = it + 1
        print('Iteração {} - Somatório: {}'.format(it, somador))
        print('_'*5)
        print(' ')
    return somador
print(vetor)
print(' ')
somatorio_quadrado(vetor)
print('-'*50)
print(' ')

##### Questão 1.i #####
print('i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor')
print('Resposta:')
pos = 0
while vetor[pos] != 8:
    print(vetor[pos])
    pos = pos + 1
    if pos == (len(vetor)):
        print('Quando a posição {} é acionada, a condição estabelecida retorna true com isso a funçao while é desabilitada'.format(pos))
        break
print('-'*50)
print(' ')

##### Questão 1.j #####
print('j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.')
print('Resposta:')
print('Sequência de valores: {}'.format(list(range(1, len(vetor)+1, 1))))
print('-'*50)
print(' ')

##### Questão 1.k #####
print('h) Concatene o vetor da letra a com o vetor da letra j.')
print('Resposta:')
q1k = list(range(1, len(vetor)+1, 1))
vetores_concatenados = np.concatenate((vetor, q1k), axis=0)
print(vetores_concatenados)
print('-'*50)
print(' ')


##### Número do exercício #####
print('EXERCÍCIO 02')

##### Questão 2.a #####
print('a) Declare a matriz abaixo com a biblioteca numpy.')
print(' '*50, '1 3 22')
print(' '*50, '2 8 18')
print(' '*50, '3 4 22')
print(' '*50, '4 1 23')
print(' '*50, '5 2 52')
print(' '*50, '6 2 18')
print(' '*50, '7 2 25')
print('Resposta:')
matriz = np.array([[1, 3, 22], [2, 8, 18], [3, 4, 22], [4, 1, 23], [5, 2, 52], [6, 2, 18], [7, 2, 25]])
print('Matriz:')
print(matriz)
print('-'*50)
print(' ')

##### Questão 2.b #####
print('b) Obtenha o número de linhas e de colunas desta matriz')
print('Resposta:')
nl, nc = np.shape(matriz)
print('Número de linhas da matriz: {}'.format(nl))
print('Número de colunas da matriz: {}'.format(nc))
print('-'*50)
print(' ')

##### Questão 2.c #####
print('c) Obtenha as médias das colunas 2 e 3.')
print('Resposta:')
submatriz_col2 = matriz[:,1]
media_matriz_col2 = np.average(submatriz_col2)
print('Matriz considerando apenas coluna 2: {} e sua média é {}'.format(submatriz_col2, media_matriz_col2))
submatriz_col3 = matriz[:,2]
media_matriz_col3 = np.average(submatriz_col3)
print('Matriz considerando apenas coluna 3: {} e sua média é {}'.format(submatriz_col3, media_matriz_col3))
print('-'*50)
print(' ')

##### Questão 2.d #####
print('d) Obtenha as médias das linhas considerando somente as colunas 2 e 3')
print('Resposta:')
submatriz_l1 = matriz[0,[1,2]]
media_matriz_l1 = np.average(submatriz_l1)
print('A média da linha 1 considerando coluna 2 e 3: {}'.format(media_matriz_l1))
submatriz_l2 = matriz[1,[1,2]]
media_matriz_l2 = np.average(submatriz_l2)
print('A média da linha 2 considerando coluna 2 e 3: {}'.format(media_matriz_l2))
submatriz_l3 = matriz[2,[1,2]]
media_matriz_l3 = np.average(submatriz_l3)
print('A média da linha 3 considerando coluna 2 e 3: {}'.format(media_matriz_l3))
submatriz_l4 = matriz[3,[1,2]]
media_matriz_l4 = np.average(submatriz_l4)
print('A média da linha 4 considerando coluna 2 e 3: {}'.format(media_matriz_l4))
submatriz_l5 = matriz[4,[1,2]]
media_matriz_l5 = np.average(submatriz_l5)
print('A média da linha 5 considerando coluna 2 e 3: {}'.format(media_matriz_l5))
submatriz_l6 = matriz[5,[1,2]]
media_matriz_l6 = np.average(submatriz_l6)
print('A média da linha 6 considerando coluna 2 e 3: {}'.format(media_matriz_l6))
submatriz_l7 = matriz[6,[1,2]]
media_matriz_l7 = np.average(submatriz_l7)
print('A média da linha 7 considerando coluna 2 e 3: {}'.format(media_matriz_l7))
print('-'*50)
print(' ')

##### Questão 2.e #####
print('e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.')
print('Resposta:')
col_2 = (matriz[:, 1])
notas_menor5 = np.where(col_2<5)
print('As posições na matriz dos genótipos que possuem notas inferiores a 5 são: {}'.format(notas_menor5[0]))
bol_notas_menor5 = col_2<5
col_1 = (matriz[:, 0])
genotipos_notas_menor5 = col_1[bol_notas_menor5]
print('Os genótipos com notas inferiores a 5 são: {}'.format(genotipos_notas_menor5))
print('-'*50)
print(' ')

##### Questão 2.f #####
print('f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior ou igual a 22.')
print('Resposta:')
col_3 = (matriz[:, 2])
peso_maior21 = np.where(col_3>=22)
print('As posições na matriz dos genótipos que possuem peso de 100 grãos superior ou igual a 22: {}'.format(peso_maior21[0]))
bol_peso_maior21 = col_3>=22
col_1 = (matriz[:, 0])
genotipos_peso_maior21 = col_1[bol_peso_maior21]
print('Os genótipos com peso de 100 grãos superior ou igual a 22: {}'.format(genotipos_peso_maior21))
print('-'*50)
print(' ')

##### Questão 2.g #####
print('g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100grãos igual ou superior a 22.')
print('Resposta:')
notas_menor4 = np.where(col_2<4)
bol_notas_menor4 = col_2<4
print('As posições na matriz dos genótipos que possuem nota inferior ou igual a 3 são, {}, e as posições na matriz dos genótipos com peso de 100 grãos superior ou igual a 22: {}'.format(notas_menor4[0], peso_maior21[0]))
genotipos_q2g = col_1[bol_peso_maior21 & bol_notas_menor4]
print('Os genótipos com peso de 100 grãos superior ou igual a 22 e nota de severidade de doença inferior ou igual a 3: {}'.format(genotipos_q2g))
print('-'*50)
print(' ')

##### Questão 2.h #####
print('h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da matriz e o seu respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido. Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z". Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25')
print('Resposta:')
nl, nc = np.shape(matriz)
print(matriz)
print(' ')
print('Número de Linhas: {}'.format(nl))
print('Número de Colunas: {}'.format(nc))
print (' ')
contador = 0
genot = []
for i in np.arange(0,nl,1):
    if matriz[i, 2] > 24:
        genot.append(matriz[i, 0])
    for j in np.arange(0,nc,1):
        contador = contador + 1
        print('Iteração {} --> Na linha {} e na coluna {} ocorre o valor {}'.format(contador, i, j, matriz[int(i),int(j)]))
print(' ')
print ("A partir destes dados, os genótipos com peso de 100 grãos iguais ou superiores a 25 foram: {}".format(genot))
print('-'*50)
print(' ')

##### Número do exercício #####
print('EXERCÍCIO 03')

##### Questão 3.a #####
print('a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer, baseada em um loop (for).')
print('Resposta:')
print('Foi criado uma função no arquivo REO1Questão3.py')
from REO1Questão3 import media, var_amostral
print('-'*50)
print(' ')

##### Questão 3.b #####
print('b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal com média 100 e variância 2500. Pesquise na documentação do numpy por funções de simulação.')
print('Resposta:')
vetor10 = np.random.normal(100,50,10)
vetor100 = np.random.normal(100,50,100)
vetor1000 = np.random.normal(100,50,1000)
print('Para não ficar muito extenso a resposta dessa questão, deixarei apenas o vetor10 com 10 valores disponível visível')
print('Vetor 10:')
print(vetor10)
print('-'*50)
print(' ')

##### Questão 3.c #####
print('c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.')
print('Resposta:')
print('Vetor 10: média {}, variância {}'.format(media(vetor10), var_amostral(vetor10)))
print('Vetor 100: média {}, variância {}'.format(media(vetor100), var_amostral(vetor100)))
print('Vetor 1000: média {}, variância {}'.format(media(vetor1000), var_amostral(vetor1000)))
print('-'*50)
print(' ')

##### Questão 3.d #####
print('d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000')
print('Resposta:')
plt.style.use('dark_background')
count, bins, ignored = plt.hist(vetor10, 30, density=True)
plt.plot(bins, 1/(50 * np.sqrt(2 * np.pi)) *
np.exp( - (bins - 100)**2 / (2 * 50**2) ),linewidth=5, color='b')
plt.title('Vetor 10')
plt.show()
count, bins, ignored = plt.hist(vetor100, 30, density=True)
plt.plot(bins, 1/(50 * np.sqrt(2 * np.pi)) *
np.exp( - (bins - 100)**2 / (2 * 50**2) ),linewidth=5, color='y')
plt.title('Vetor 100')
plt.show()
count, bins, ignored = plt.hist(vetor1000, 30, density=True)
plt.plot(bins, 1/(50 * np.sqrt(2 * np.pi)) *
np.exp( - (bins - 100)**2 / (2 * 50**2) ),linewidth=5, color='g')
plt.title('Vetor 1000')
plt.show()

vetor100000 = np.random.normal(100,50,100000)
count, bins, ignored = plt.hist(vetor100000, 30, density=True)
plt.plot(bins, 1/(50 * np.sqrt(2 * np.pi)) *
np.exp( - (bins - 100)**2 / (2 * 50**2) ),linewidth=5, color='r')
plt.title('Vetor 100000')
plt.show()
print('Gráficos dos vetores 10, 100, 1000 e 100000 foram gerados, respectivamente')
print('-'*50)
print(' ')

print('EXERCÍCIO 04')

##### Questão 4.a #####
print('a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto a quatro variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca numpy, apresente os dados e obtenha as informações de dimensão desta matriz.')
print('Resposta:')
dados = np.loadtxt('dados.txt')
print('Dados de avaliação dos genótipos:')
print('   Gen     Rep  v1     v2     v3     v4     v5')
print(dados)
print(' ')
nl_q4, nc_q4 = dados.shape
print('Número de linhas: {}'.format(nl_q4))
print('Número de colunas: {}'.format(nc_q4))
print(' ')
tipo_dados = type(dados)
print('Tipo da matriz: {}'.format(tipo_dados))
print('-'*50)
print(' ')

##### Questão 4.b #####
print('b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy')
print('Resposta:')
print('--> Função np.unique:')
print(' ')
help(np.unique)
print('-'*25)
print(' ')
print('--> Função np.where:')
print(' ')
help(np.where)
print('-'*50)
print(' ')

##### Questão 4.c #####
print('c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas')
print('Resposta:')
print("Genótipos: {}".format(np.unique(dados[:, 0])))
print("Repetições: {}".format(np.unique(max(dados[:, 1]))))
print('-'*50)
print(' ')

##### Questão 4.d #####
print('d) Apresente uma matriz contendo somente as colunas 1, 2 e 4')
print('Resposta:')
print('Matriz contendo apenas as colunas 1, 2 e 4:')
print('   Gen    Rep   v2')
print(dados[:, [0, 1, 3]])
print('-'*50)
print(' ')

##### Questão 4.e #####
print('e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel da coluna 4. Salve esta matriz em bloco de notas.')
print('Resposta:')
dados_q4e = dados[:, [0, 1, 3]]
matriz0 = np.zeros((len(np.unique(dados_q4e[:, 0])), 5))
it = 0
for n in range(0, len(np.unique(dados_q4e[:, 0])), 1):
    it = it + 1
    print(' ')
    print('Genótipo: {}'.format(it))
    print('Máxima: {}'.format(np.max((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2])))
    print('Mínimo: {}'.format(np.min((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2])))
    print('Média: {}'.format((np.around(np.mean((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2]), 4))))
    print('Variância: {}'.format((np.around(np.var((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2]), 4))))
    matriz0[n, 0] = n + 1
    matriz0[n, 1] = np.max((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2])
    matriz0[n, 2] = np.min((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2])
    matriz0[n, 3] = np.around(np.mean((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2]), 4)
    matriz0[n, 4] = np.around(np.var((dados_q4e[dados_q4e[:, 0] == n + 1])[:, 2]), 4)
    print('-' * 25)
np.savetxt('Resultados_da_Matriz.txt', matriz0)
print(' ')
print('-' * 50)
print('Matriz Gerada --> Genótipo, Máxima, Mínima, Média e Variância')
print(matriz0)
print(' ')
print('-' * 50)

##### Questão 4.f #####
print('f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz gerada na letra anterior.')
print('Resposta:')
media_sup_500 = matriz0[:, 3] > 499
sup_500 = matriz0[media_sup_500]
print("Os genótipos que possuem média igual ou superior a 500 da matriz da Questão_4.E são: {} e {}".format(sup_500[:, 0][0], sup_500[:, 0][1]))
print('-'*50)
print(' ')

##### Questão 4.g #####
print('g) Apresente os seguintes graficos:')
print('- Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para mostrar mais de um grafico por figura')
print('Resposta:')
matriz0_10x6 = np.zeros((len(np.unique(dados[:, 0])), 6))
it = 0
for i in range(0, len(np.unique(dados[:, 0])), 1):
    it = it + 1
    matriz0_10x6[i, 0] = i + 1
    matriz0_10x6[i, 1] = np.around(np.mean((dados[dados[:, 0] == i + 1])[:, 2]), 4)
    matriz0_10x6[i, 2] = np.around(np.mean((dados[dados[:, 0] == i + 1])[:, 3]), 4)
    matriz0_10x6[i, 3] = np.around(np.mean((dados[dados[:, 0] == i + 1])[:, 4]), 4)
    matriz0_10x6[i, 4] = np.around(np.mean((dados[dados[:, 0] == i + 1])[:, 5]), 4)
    matriz0_10x6[i, 5] = np.around(np.mean((dados[dados[:, 0] == i + 1])[:, 6]), 4)
print('Levando em conta que a 1ªcol genótipos, 2ªcol V1, 3ªcol V2, 4ªcol V3, 5ªcol V4 e 6ªcol V5')
print(matriz0_10x6)
print(' ')
print('Gráfico requisitado gerado em uma janela separada')

plt.figure('Média das variáveis')
plt.subplot(3, 2, 1)
plt.grid(True)
plt.bar(x=matriz0_10x6[:, 0], height=matriz0_10x6[:, 1])
plt.title('V1', loc='center')
plt.xticks(matriz0_10x6[:, 0])
plt.ylabel("Média")

plt.subplot(3, 2, 2)
plt.grid(True)
plt.bar(x=matriz0_10x6[:, 0], height=matriz0_10x6[:, 2])
plt.title('V2', loc='center')
plt.xticks(matriz0_10x6[:, 0])

plt.subplot(3, 2, 3)
plt.grid(True)
plt.bar(x=matriz0_10x6[:, 0], height=matriz0_10x6[:, 3])
plt.title('V3', loc='center')
plt.ylabel("Média")
plt.xticks(matriz0_10x6[:, 0])

plt.subplot(3, 2, 4)
plt.grid(True)
plt.bar(x=matriz0_10x6[:, 0], height=matriz0_10x6[:, 4])
plt.title('V4', loc='center')
plt.xticks(matriz0_10x6[:, 0])

plt.subplot(3, 2, 5)
plt.grid(True)
plt.bar(x=matriz0_10x6[:, 0], height=matriz0_10x6[:, 5])
plt.title('V5', loc='center')
plt.xticks(matriz0_10x6[:, 0])
plt.ylabel("Média")
plt.show()
print('-'*50)
print(' ')
print('- Disperão 2D da médias dos genótipos (Utilizar as três primeiras variáveis). No eixo X uma variável e no eixo Y outra.')
print('Resposta:')
print('Gráfico requisitado gerado em uma janela separada')
plt.style.use('ggplot')
plt.figure('2D Scatter Graph')

plt.subplot(1, 3, 1)
for i in np.arange(0, len(np.unique(dados[:, 0])), 1):
    plt.scatter(matriz0_10x6[i, 1], matriz0_10x6[i, 2])
plt.xlabel('V1')
plt.ylabel('V2')

plt.subplot(1, 3, 2)
for i in np.arange(0, len(np.unique(dados[:, 0])), 1):
    plt.scatter(matriz0_10x6[i, 1], matriz0_10x6[i, 3])
plt.xlabel('V1')
plt.ylabel('V3')

plt.subplot(1, 3, 3)
for i in np.arange(0, len(np.unique(dados[:, 0])), 1):
    plt.scatter(matriz0_10x6[i, 1], matriz0_10x6[i, 4])
plt.xlabel('V1')
plt.ylabel('V4')

plt.show()
print('-'*50)
print(' ')
