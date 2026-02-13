#!/usr/bin/env python3
import numpy as np
import gurobipy as gp
#import cvxpy as cp
import time
import random
from auxiliary_functions import *

def Feasibility_Entropic_vector(A, E, C, d):

    b = np.zeros((len(A)))
    e = np.zeros((len(E)))
    prob = gp.Model()
    
    h = prob.addMVar(len(A[0]), lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY )
    
    prob.setObjective(0, gp.GRB.MINIMIZE)
    #prob.setObjective(C@h, gp.GRB.MAXIMIZE)

    #print('Basicas')
    prob.addConstr((A @ h) <= b)
    #print('RICS')
    prob.addConstr((E @ h) == e)
    #print('Box')
    prob.addConstr((C@h) == d)
    
    prob.update()
    prob.setParam('OutputFlag',False)
    prob.setParam("BarHomogeneous", 1)# Including solve methods useful for infeasibility problems
    prob.optimize()
    
    return prob.Status

def vector_H(M, pabc):
    h = []
    for k in M:
        p = marginalizacao(pabc, k)
        h = h+[H(p)]
    return np.array(h)

def Equalities_spiral_inflation(vetor_H):
    # 0,  1,  2,  3,  4,  5
    # A0, B0, C0, A1, B1, C1

    matrix_E = np.zeros((1,len(vetor_H)), dtype='int') # Matriz de desigualdades basicas com linhas a serem incrementadas pela newrow conforme necessidade
    newrow=np.zeros(len(vetor_H), dtype='int')

    
    ###############################################################################################################
    #        INDEPENDENCE 
    ###############################################################################################################
    
    matrix_E[len(matrix_E)-1][vetor_H.index((3,4,5))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((3,))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((4,))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((5,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])
    
    matrix_E[len(matrix_E)-1][vetor_H.index((2,3,4))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((2,3))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((4,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])
    
    matrix_E[len(matrix_E)-1][vetor_H.index((0,4,5))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((0,4))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((5,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])
    
    matrix_E[len(matrix_E)-1][vetor_H.index((1,3,5))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((1,5))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((3,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])
    

    
    
    matrix_E[len(matrix_E)-1][vetor_H.index((0,5))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((0,))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((5,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((1,3))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((1,))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((3,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])
    
    matrix_E[len(matrix_E)-1][vetor_H.index((2,4))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((2,))] = -1
    matrix_E[len(matrix_E)-1][vetor_H.index((4,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])
    
    ###############################################################################################################
    #        CONSISTENCY 
    ###############################################################################################################
    
    matrix_E[len(matrix_E)-1][vetor_H.index((0,))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((3,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((1,))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((4,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((2,))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((5,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])
    
    matrix_E[len(matrix_E)-1][vetor_H.index((0,4))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((0,1))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((1,5))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((1,2))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((2,3))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((0,2))] = -1
    
    
    #matrix_E=np.vstack((matrix_E,-matrix_E))
    return matrix_E




def Consistency_complete_spiral(vetor_H):
# 0,  1,  2,  3,  4,  5 , 6 , 7 , 8 , 9 , 10, 11
# A0, B0, C0, A1, B1, C1, a0, a1, g0, g1, b0, b1 
    matrix_E = np.zeros((1,len(vetor_H)), dtype='int') # Matriz de desigualdades basicas com linhas a serem incrementadas pela newrow conforme necessidade
    newrow=np.zeros(len(vetor_H), dtype='int')

    
    
    matrix_E[len(matrix_E)-1][vetor_H.index((6,))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((7,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((8,))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((9,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((10,))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((11,))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    
    
    matrix_E[len(matrix_E)-1][vetor_H.index((0,10))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((3,10))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((0,8))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((3,9))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((0,8,10))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((3,9,10))] = -1
    matrix_E = np.vstack([matrix_E, newrow])


    
    matrix_E[len(matrix_E)-1][vetor_H.index((1,8))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((4,8))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((1,6))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((4,7))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((1,6,8))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((4,7,8))] = -1
    matrix_E = np.vstack([matrix_E, newrow])



    matrix_E[len(matrix_E)-1][vetor_H.index((2,6))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((5,6))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((2,10))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((5,11))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((2,6,10))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((5,6,11))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    
    
    matrix_E[len(matrix_E)-1][vetor_H.index((0,1,6,8))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((0,4,7,8))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((1,2,6,10))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((1,5,6,11))] = -1
    matrix_E = np.vstack([matrix_E, newrow])

    matrix_E[len(matrix_E)-1][vetor_H.index((0,2,8,10))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((2,3,9,10))] = -1


    
    #matrix_E=np.vstack((matrix_E,-matrix_E))
    return matrix_E
