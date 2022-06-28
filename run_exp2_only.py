import gc
import sys
import os
#launcher for experiments
# syntax 2 0.01 1024 4 3
# Dataset , selectivity, datasize, cluster count, datatype
# dataset (0 uniform, 1 1cluster, 2 multicluster)
# selectivity % of 1 bits in mask [0 ; 1]
# datasize in MIB of input col
# datatypes 1-uint8 2-uint16 3-uint32 4-int 5-float 6-double
# algos (1: cub + space 8, 0: all)

def run_sel(sel,dataset, datatype, cluster, device, algos=1):
    run=(str(dataset)+" "+str(sel)+" "+str(1024)+" "+str(cluster)+" "+str(datatype)+" "+str(device)+" "+str(algos))
    #print(run)
    #cmd = './build/gpu_compressstore2 '+run
    #in debug folder
    cmd = './gpu_compressstore2 ' + run
    os.system(cmd)



def run_exp2(device="exp2",sel_increment=4):

    sel = 0
    dataset = 0
    max_cluster = 32
    for k in range(3,4,1):
        datatype = k
        for f in range(1, 100, sel_increment):
            sel=f/100
            for i in range(0, 3, 1):
                    dataset = i
                    c = 1
                    if dataset == 2:
                        while (c <= max_cluster):
                            run_sel(sel, dataset, datatype, c, device,0)
                            c = c * 2
                    else:
                        run_sel(sel, dataset, datatype, 1, device,0)


if __name__ == '__main__':
   
    print("running experiment 2.....")
    run_exp2(sel_increment=4)
    print("done")
   