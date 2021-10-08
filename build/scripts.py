#!/usr/bin/env python3
# nohup python3 scripts.py </dev/null &>/dev/null &
# ps -ef | grep ../bin_SI/annealer_gpu_SI
import os
import time
import subprocess

dict_Lattice = {4: ("-a ../bin_SI/J_Matrix_4x4.txt ","-n 35 ", "-m 1000 -x 7.2 -y 0.001 -d "), 
                  20 : ("-a ../bin_SI/J_Matrix_20x20.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d "),
                    40 : ("-a ../bin_SI/J_Matrix_40x40.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d "),
                      60 : ("-a ../bin_SI/J_Matrix_60x60.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d ")}
'''
4: ("-a ../bin_SI/J_Matrix_4x4.txt ","-n 35 ", "-m 1000 -x 7.2 -y 0.001 -d "), 
                  20 : ("-a ../bin_SI/J_Matrix_20x20.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d "),
                    40 : ("-a ../bin_SI/J_Matrix_40x40.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d "),
                      60 : ("-a ../bin_SI/J_Matrix_60x60.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d "),
                        80 : ("-a ../bin_SI/J_Matrix_80x80.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d "),
                         100 : ("-a ../bin_SI/J_Matrix_100x100.txt ","-n 35 ", "-m 6000 -x 7.2 -y 0.001 -d "),
                           120 : ("-a ../bin_SI/J_Matrix_120x120.txt ","-n 35 ", "-m 10000 -x 7.2 -y 0.001 -d "),          
                            140 : ("-a ../bin_SI/J_Matrix_140x140.txt ","-n 35 ", "-m 12000 -x 7.2 -y 0.001 -d "),
                              180 : ("-a ../bin_SI/J_Matrix_180x180.txt ","-n 35 ", "-m 14000 -x 7.2 -y 0.001 -d "),
                                230 : ("-a ../bin_SI/J_Matrix_230x230.txt ","-n 35 ", "-m 20000 -x 7.2 -y 0.001 -d ")
'''

# to check type "ps -ef | grep '../bin_SI/annealer_gpu_SI"
def executeCpp(lattice_num : int):
    lattice_cmd = dict_Lattice[lattice_num]
    cmd = "../bin_SI/annealer_gpu_SI "
    last_part_cmd = "< /dev/null &> /dev/null"
    cmd = cmd + lattice_cmd[0] + lattice_cmd[1] + lattice_cmd[2] + last_part_cmd 
    s = subprocess.check_output(cmd, shell = True)
  
def executeCppBad():
  
    # create a pipe to a child process
    data, temp = os.pipe()
  
    # write to STDIN as a byte object(convert string
    # to bytes with encoding utf8)
    os.write(temp, bytes("5 10\n", "utf-8"));
    os.close(temp) # STDIN=temp, shell=True);
  
    # store output of the program as a byte string in s
    s = subprocess.check_output("nohup ../bin_SI/annealer_gpu_SI ../bin_SI/J_Matrix_4x4.txt 3 1000 < /dev/null &> /dev/null", shell = True)
    #subprocess.call(["nohup ../bin_SI/annealer_gpu_SI ../bin_SI/J_Matrix_180x180.txt 3 1 < /dev/null &> /dev/null"])
    breaksearch = True
    while breaksearch:
        time.sleep(100)
        cppProcesses = subprocess.check_output("ps -ef | grep '../bin_SI/annealer_gpu_SI ../bin_SI/J_Matrix_4x4.txt 3 1000'",shell=True).decode()
        cppProcesses = cppProcesses.split('\n')
        for cppProcess in cppProcesses:
            found = cppProcess.find(grep)
            if found == -1:
                breaksearch = False                                        
          
    # decode s to a normal string
    print(s.decode("utf-8"))

# Driver function
if __name__=="__main__":  
    for key in dict_Lattice.keys():
        executeCpp(key)    
    