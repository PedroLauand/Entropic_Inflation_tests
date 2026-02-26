import numpy as np
from auxiliary_functions import vector_H
from Feas_problem import Feasibility_Entropic_vector

def bisection(p, M, d, basicas, Equalities, C):

    pw = (1/(d*d*d))*np.ones((d,d,d), dtype='float')

#-----------PARAMETROS BISECCAO----------------------------------------------------------
    max_it = 15
    aux = 0
    iterac = 0
    gammaB = 0.0
    atol=10**-4
    bound = 0.0

    gammaU = 1- 10**-16
    gamma = gammaB

    while(aux < max_it ):
        Viola = False
        iterac = iterac +1
        pabc = gamma*p+ (1-gamma)*pw
        H = vector_H(M, pabc)

        status = Feasibility_Entropic_vector(basicas, Equalities, C, H)


        if status == 3:
			#print('########################################################################')
            print(status, gamma)
            gammaU = gamma
            if(abs(gammaU - gammaB) > atol):
                gamma = gammaU - (gammaU - gammaB)/2
            else:
                break
        else:
			#print('########################################################################')
            print(status, gamma)
            gammaB = gamma
            if(abs(gammaU - gammaB) > atol):
                gamma = gammaB + (gammaU - gammaB)/2
            else:
                break
        aux = aux +1
    #print('Gamma U: ',gammaU)
    # The last gamma violating the entropics
    return gammaU