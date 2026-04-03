import itertools as it
import numpy as np
import math

cardinalities = 4*[2]
Possibilities = list(it.product(*[range(i) for i in cardinalities]))
def Index222(variables):
	return Possibilities.index(variables)

def p222():
    p = np.zeros((2,2,2), dtype='float')
    p[0][0][0] = 1/2
    p[1][1][1] = 1/2
    return p

def pD():
    p = np.zeros((2,2,2), dtype='float')
    p[0][0][0] = 1
    return p

def pW():
    p = np.zeros((2,2,2), dtype='float')
    p[0][0][1] = 1/3
    p[1][0][0] = 1/3
    p[0][1][0] = 1/3
    return p

#------------FUNÇÃO QUE CALCULA A ENTROPIA------------------------------------------
def H(prob):
	S=np.size(prob)# Numero de elementos
	Pro=np.reshape(prob, S) #concatenando todos os elementos num vetor
	h=np.zeros(S, dtype='float')
	for i in range(0,S): #Calcuando entropia para as probabilidades não nulas
		if Pro[i]>=10**-16:
			#print('pro',Pro[i])
			h[i]=-Pro[i]*math.log(Pro[i],2)
	Ent=h.sum() # Entropia de Shannon
	return Ent

def marginalizacao(P, marg):
	n =P.ndim
	x=np.arange(n)
	
	elim = np.delete(x, marg)
	elim = np.sort(elim)[::-1].astype(int)
	#print(elim)
	
	for i in elim:
		P = P.sum(axis=i,dtype='float')
	
	return P