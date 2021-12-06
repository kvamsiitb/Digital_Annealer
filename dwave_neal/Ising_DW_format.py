'''
Note: The file format required for GPU based Ising Solver is a "Upper Triangular matrix"
 with all diagonal elements are zero
For example:

    N = 100 # Number of variables
    J = np.random.rand(N,N)
    J = np.triu(J, 1) # We only consider upper triangular matrix ignoring the diagonal
    h = np.random.rand(N,1)

    print(J) # is the required format for the input of ising machine
    print(h) # is the required format for the input of ising machine
 '''
import numpy as np
from numpy.random import rand

def readTxtFile(filename:str):
    num_spins = 0
    with open( filename, "r") as f:
        line = f.readline()
        meta_data = line.split()
        num_spins = int(meta_data[0])
        iters = int(meta_data[1])
        J_mat = np.zeros( ( num_spins, num_spins ) )
        index = 0
        while index < iters :
            line = f.readline()
            meta_data = line.split()
            try:
                J_mat[int(meta_data[0]) - 1][ int( meta_data[1] ) - 1 ] = float(meta_data[2])
            except:
                pass
            index = index + 1
    return J_mat,  num_spins      

def readLinTxtFile(filename:str):
    num_spins = 0
    linear_vec = []
    with open( filename, "r") as f:
        line = f.readline()
        meta_datas = line.split()
        for meta in meta_datas:
            linear_vec.append(float(meta))

    return linear_vec

def main():
    J, num_spins = readTxtFile("coupling_6pt.txt")
    h = readLinTxtFile('linear_6pt.txt')

    dict_h = {}
    dict_J = {}
    
    for i in enumerate(h):
      dict_h[i[0]] = i[1]
    print(dict_J)
    print(dict_h)
    for ii in range(J.shape[0]):
      for jj in range(J.shape[1]):
        if J[ii][jj]:   
          a_tup = (ii, jj)
          dict_J[a_tup] = J[ii][jj]

    print(dict_J)
    print(dict_h)
main()
