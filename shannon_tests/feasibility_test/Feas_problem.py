#!/usr/bin/env python3
import numpy as np
import gurobipy as gp
#import cvxpy as cp
import time
import random
from auxiliary_functions import *

def Feasibility_Entropic_vector(A, C, d):

    b =np.zeros((len(A))) 
    prob = gp.Model()
    
    h = prob.addMVar(len(A[0]), lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY )
    
    prob.setObjective(0, gp.GRB.MINIMIZE)
    
    prob.addConstr((A @ h) <= b)    
    prob.addConstr((C@h) == d)
    
    prob.update()
    prob.setParam('OutputFlag',False)
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
    
    # NOTE: Unusual equality (B0,A1,C1) = (B0,A1) + (A1)
    # This enforces H(B0,A1,C1) - H(B0,A1) - H(A1) = 0.
    # It is stronger than standard independence with C1, and is not imposed
    # in the spiral LP code. See status report for details.
    matrix_E[len(matrix_E)-1][vetor_H.index((1,3,5))] = 1
    matrix_E[len(matrix_E)-1][vetor_H.index((1,3))] = -1
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
    #matrix_E = np.vstack([matrix_E, newrow])
    
    
    matrix_E=np.vstack((matrix_E,-matrix_E))
    return matrix_E

