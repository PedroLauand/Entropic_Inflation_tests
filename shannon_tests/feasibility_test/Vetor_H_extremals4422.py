import random
import math
import numpy as np

from auxiliary_functions import *
from desc_entro.desc_entro import cenario_marginal



def vetor_H_RAC21(M,opt_func, paBDxy):

    Px0x1aalB0 = np.zeros((2, 2, 2, 2, 2), dtype='float')
    Px0x1aalB1 = np.zeros((2, 2, 2, 2, 2), dtype='float')

    paBDxy = PRab()

    #(p1,p2,p3,p4)=(0,1,0,1)
    (p1,p2,p3,p4)=(1,1,1,1)
    #print(p1,p2,p3,p4)
    px=(1/(p1+p2+p3+p4))*np.array([p1,p2,p3,p4]) 
    X = {(0,0):0,(0,1):1,(1,0):2,(1,1):3}
    Y = {(0,0):0,(0,1):1,(1,0):2,(1,1):3}
    
    channelpre = 1
    for pc in np.arange(0, 0.5, 0.5/channelpre):
        pc = 0.0
        #print(pc)
        pmlDm = np.array([[(1-pc), pc],[pc, (1-pc)]]) # Channel
        #print(pmlDm)
        #-----------PROBABILIDADE CONJUNTA---------------------------------------------------------
        
        for x0, x1, a, ml, B in it.product(range(2),range(2),range(2),range(2),range(2)):
            
            Px0x1aalB0[x0][x1][a][al][B] = (1/4)*paBDxy[a][B][X[(x0,x1)]][Y[(al,0)]]*pmlDm[al][a]
            Px0x1aalB1[x0][x1][a][al][B] = (1/4)*paBDxy[a][B][X[(x0,x1)]][Y[(al,1)]]*pmlDm[al][a]
            
        
		#---------- ENTROPIC VECTOR---------------------------------------------------------------------
        h = np.array([len(M)+1], dtype ='float')# Condicao para primeiro elemento. Por algum motivo e boa 
        
        for k in M:
            if (k.count(5)== 0): #Se tiver g1 usa uma distibuicao, se tiver g0 usa outra
                p = marginalizacao(Px0x1aalB0, k)
                #print(p)
                if(h[0]==len(M)+1): #Apenas condicao paa ser primeiro
                	h[0] = H(p)
                	#print(k,H(p))
                else:
                	h = np.concatenate((h, np.array([H(p)])))
                	#print(k,H(p))
            else:
                #Alterando 5 por 3 para ajustar o indice da distribuicao
                j = list(k)
                j[j.index(5)] = 4
                j = tuple(j)
                #print(j)
                p = marginalizacao(Px0x1aalB1, j)
                #print(p)
                if(h[0]==len(M)+1):#Apenas condicao paa ser primeiro
                	h[0] = H(p)
                    #print(k,H(p))	
                else:
                    h = np.concatenate((h, np.array([H(p)])))
                    #print(k,H(p),p)
                    
        #-----------------VALUE FOR THE OPTIMIZATION FUCTION-----------------------
        h_aa = H(marginalizacao(Px0x1aalB0, opt_func[0]))+H(marginalizacao(Px0x1aalB0, opt_func[1]))-H(marginalizacao(Px0x1aalB0, opt_func[2]))
        #print(H(marginalizacao(Px0x1mmlG0, (2,))), H(marginalizacao(Px0x1mmlG0, (3,))))
    return h, h_aa
