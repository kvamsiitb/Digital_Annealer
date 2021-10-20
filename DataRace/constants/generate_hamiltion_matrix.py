# create J_ij matrix

lattice_width = 20
filename = "J_Matrix_{}x{}.txt".format(lattice_width, lattice_width)
f = open(filename, "w")

# width of the lattice and total number of spins equal to lattice_width^2
list_connection = []
for i in range(lattice_width):
    for j in range(lattice_width):
        if i != 0:
            list_connection.append(
                (i * lattice_width + j + 1, (i - 1) * lattice_width + j + 1, 1)
            )
        if j != 0:
            list_connection.append(
                (i * lattice_width + j + 1, i * lattice_width + (j - 1) + 1, 1)
            )
        if j != lattice_width - 1:
            list_connection.append(
                (i * lattice_width + j + 1, i * lattice_width + (j + 1) + 1, 1)
            )
        if i != lattice_width - 1:
            list_connection.append(
                (i * lattice_width + j + 1, (i + 1) * lattice_width + j + 1, 1)
            )

num_spins = lattice_width ** 2
print(lattice_width, num_spins, len(list_connection))
print(list_connection)

f.write("{} {}\n".format(num_spins, len(list_connection)))
for ui in list_connection:
    f.write("{} {} {}\n".format(ui[0], ui[1], ui[2]))
f.close()
