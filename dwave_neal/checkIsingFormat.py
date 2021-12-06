'''
Purpose: Check the file format, "UPPER TRIANGULAR" Matrix for Jij/ coupling matrix

'''


import numpy as np

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



def main():
    print("check file format of coupling matrix")
    J, num_spins = readTxtFile("J_Matrix_800.txt")
    print(J)
    print( np.allclose(J, np.triu(J)) ) # check if upper triangular

if __name__ == '__main__' :
    main()
