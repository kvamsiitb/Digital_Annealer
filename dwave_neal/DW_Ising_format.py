'''
Note:
Qubo is converted to Ising model but still the matrix is not upper triangular matrix.

[1] step convert generic matrix to an upper triangular matrix

Output: 2 files with one containing  the coupling constant and other the linear terms as per
GPU Ising solver format.

Note: Converts DWave QUBO dictionary into 2 files - one with Ising coupling coefficients and the other with the Ising linear terms (external field terms in the model).
 '''


from dimod.utilities import qubo_to_ising

def QUBO_to_files(Q):
    [l,J,o] = qubo_to_ising(Q) # if you already have variables in Ising form you can comment out this line and change the function from QUBO_to_files(Q) to QUBO_to_files(l,J)
    dict_J=defaultdict()
    for i in J.keys():
        if (i[1],i[0]) not in dict_J.keys():
            dict_J[i]=J[i]
        else:
            dict_J[(i[1],i[0])]+=J[i] # this step is necessary to ensure DWave and the files generated give the same energy for any given configuration.
    h=l
    N = len(x)**2
    file1 = open("myfile.txt", "w")
    file1.write(str(N) + ' ' + str(len(dict_J))+ '\n')
    for i  in dict_J:
      #print(i, dict_J[i])
      file1.write(str(i[0]+1) + ' ' + str(i[1]+1) + ' '+str(-dict_J[i])+ '\n')
    file1.close()

    file2 = open("myfile2.txt", "w")
    for i  in h:
      #print(i[0])
      file2.write(str(-h[i]) +' ')
    file2.close()
