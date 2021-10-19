
#create J_ij matrix
lattice_width = 10
filename = "linear_{}x{}.txt".format(lattice_width,lattice_width)
f = open(filename, "w")


# width of the lattice and total number of spins equal to lattice_width^2
list_connection = []
'''
for i in range( lattice_width):
    for j in range(lattice_width):
            list_connection.append( 0.0 )
 '''
for i in range( lattice_width**2):
    list_connection.append( 0.0 )
    
num_spins = lattice_width** 2
print(lattice_width, num_spins)

for ui in range(len(list_connection)):
    if ui is not len(list_connection) - 1:
        f.write("{} ".format( list_connection[ui]) )
    else:
        f.write("{}\n".format( list_connection[ui]) )
f.close()
