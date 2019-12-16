# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:35:05 2016

@author: emi
"""
import subprocess

import numpy as np
import os
import math


class FilaMatrizCoOcurrencia:
    def __init__(self, idx):
        self.palabra_index = idx
        self.otrasPalabras = dict()

    def agregarOcurrencia(self, otraPalabra):
        if otraPalabra in self.otrasPalabras:
            self.otrasPalabras[otraPalabra] += 1
        else:
            self.otrasPalabras[otraPalabra] = 1

    def __getitem__(self, otraPalabra):
        if otraPalabra in self.otrasPalabras:
            return self.otrasPalabras[otraPalabra]
        else:
            return 0


def calcularOcurrencias(documentos, voca):
    matriz = []
    for j in range(len(voca)):
        matriz.append(FilaMatrizCoOcurrencia(j))
    ocurrencias = np.zeros(len(voca))

    for documento in documentos:
        for w in documento:
            idx_palabra_i = voca[w]
            ocurrencias[idx_palabra_i] = ocurrencias[idx_palabra_i] + 1
        for i in range(len(documento) - 1):
            for j in range(i, len(documento)):
                idx_palabra_j = voca[str(documento[j])]
                matriz[idx_palabra_i].agregarOcurrencia(idx_palabra_j)

    return (ocurrencias, matriz)


class ModeloBTM:
    def __init__(self, output_dir, k):
        self.output_dir = output_dir
        self.mPDocEnTopico = None
        self.k = k
        self.mPPalabraDadoTopico = None
        self.vocabulario = None

    def cantidadDeTopicos(self):
        return self.k

    def matrizProbabilidadDocumentoEnTopico(self):
        if (self.mPDocEnTopico == None):
            self.mPDocEnTopico = np.genfromtxt(self.output_dir + "model/" + "k" + str(self.k) + ".pz_d", delimiter=" ")
            # self.mPDocEnTopico = np.genfromtxt(self.output_dir + "/model/" + "k"+str(self.k)+".pz_d", delimiter=" ")
        return self.mPDocEnTopico

    def cantidadDeDocumentos(self):
        m = self.matrizProbabilidadDocumentoEnTopico()
        return m.shape[0]

    def topicoDeDocumento(self, i):
        m = self.matrizProbabilidadDocumentoEnTopico()
        return m[i, :].argmax()

    def topicoDeDocumentos(self):
        topicos = []
        for i in range(self.cantidadDeDocumentos()):
            topicos.append(self.topicoDeDocumento(i))
        return topicos

    def matrizProbabilidadPalabraDadoTopico(self):
        if (self.mPPalabraDadoTopico == None):
            self.mPPalabraDadoTopico = np.genfromtxt(self.output_dir + "model/" + "k" + str(self.k) + ".pw_z",
                                                     delimiter=" ")
            # self.mPPalabraDadoTopico = np.genfromtxt(self.output_dir + "/model/" + "k"+str(self.k)+".pw_z", delimiter=" ")
        return self.mPPalabraDadoTopico

    def getVocabulario(self):
        if (self.vocabulario == None):
            self.vocabulario = dict()
            i = 0
            with open(self.output_dir + "/voca.txt", 'r') as f:
                for linea in f:
                    row = linea.rstrip('\n').split('\t')
                    self.vocabulario[row[1]] = i
                    i = i + 1
        return self.vocabulario

    def coherencia_de_topico(self, topico, k_palabras, ocurrencias, matriz_coocurrencia):
        m = self.matrizProbabilidadPalabraDadoTopico()
        indices_palabras_mas_probables = np.argsort(m[topico, :])[-k_palabras:]
        # print(indices_palabras_mas_probables)
        suma = 0.0
        for t in range(1, k_palabras):
            indice_palabra_t = indices_palabras_mas_probables[t]
            for l in range(0, t):
                indice_palabra_l = indices_palabras_mas_probables[l]
                suma = suma + math.log(
                    (matriz_coocurrencia[indice_palabra_t][indice_palabra_l] + 1.0) / ocurrencias[indice_palabra_l])
        return suma

    def coherencia(self, k_palabras, ocurrencias, matriz_coocurrencia):
        suma_total = 0.0
        for j in range(0, self.k):
            suma_total = suma_total + self.coherencia_de_topico(j, k_palabras, ocurrencias, matriz_coocurrencia)

        return (1.0 / self.k) * suma_total


class BTM:

    def __init__(self, path, k, iteraciones, alpha=-1, beta=0.005, path_documentos='/../BTM',
                 output_dir='/../BTM'):
        if (alpha < 0):
            self.alpha = 50.0 / k
        else:
            self.alpha = alpha
        self.beta = beta
        self.k = k
        self.iteraciones = iteraciones
        self.path = path
        self.path = path + " %s %s %s %s %s %s"
        self.path_documentos = path_documentos
        if not os.path.exists(self.path_documentos):
            os.makedirs(self.path_documentos)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def entrenar(self, documentos):
        documentos = self.copiar_documentos_txt(documentos)
        proceso = subprocess.Popen([self.path % (self.k, \
                                                 self.iteraciones, \
                                                 self.alpha, \
                                                 self.beta,
                                                 self.output_dir,
                                                 documentos)],
                                                 shell=True,
                                     executable='/bin/bash')
        proceso.wait()
        return ModeloBTM(self.output_dir, self.k)

    def copiar_documentos_txt(self, documentos):
        f = open( self.output_dir + '/docs.txt', 'w')
        for documento in documentos:
            documento_unido = ' '.join([str(word) for word in documento])
            f.write(documento_unido + "\n")
        f.close()
        return self.output_dir +  '/docs.txt'


def getModelo(datos, k, iter):
    path = str( 'BTM/script/runBTM.sh')
    voca = None
    btm = BTM(path, k, iter, -1, 0.001, str( 'BTM/entrada'), str('BTM/salida/'))
    modelobtm = btm.entrenar(datos)
    return modelobtm


def main_coherencia(datos, max_k, iter=50):
    path = str( 'BTM/script/runBTM.sh')
    voca = None
    result = []
    iteraciones = iter

    for k in range(2, max_k):
        coherencia_actual = 0
        for i in range(iteraciones):
            btm = BTM(path, k, iter, -1, 0.001, str( 'BTM/entrada'), str( 'BTM/salida/'))
            modelo = btm.entrenar(datos)

            if voca == None:
                voca = modelo.getVocabulario()
                (ocu, cocu) = calcularOcurrencias(datos, voca)
            coherencia_actual = coherencia_actual + modelo.coherencia(10, ocu, cocu)
        coherencia_actual = coherencia_actual / iteraciones
        result.append((k, coherencia_actual))

    return result


def normalizeRows(a):
    return a / a.sum(axis=1)[:, np.newaxis]


#if __name__ == "__main__":
    #m = getModelo(datos, 10, 100)

    #result=main_coherencia()
    #coherencias=[]
    #for res in result:
    #    print('k: %d' % res[0])
    #    coherencias.append(res[1])
    #    print('coherencia promedio: %d' % res[1])
    #    print('-------------------------------------------')

    #plt.plot(range(2,90), coherencias)
    #plt.xlabel('k')
    #plt.ylabel('coherencia promedio')
