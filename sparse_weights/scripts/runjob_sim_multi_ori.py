import os
import socket
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random


def runjobs():
    """
        Function to be run in a Sun Grid Engine queuing system. For testing the output, run it like
        python runjobs.py --test 1
        
    """
    
    #--------------------------------------------------------------------------
    # Test commands option
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=int, default=0)
    parser.add_argument("--cluster_", help=" String", default='burg')
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    hostname = socket.gethostname()
    if 'ax' in hostname:
        cluster = 'axon'
    else:
        cluster = str(args["cluster_"])
    
    if (args2.test):
        print ("testing commands")
    
    #--------------------------------------------------------------------------
    # Which cluster to use

    
    if platform=='darwin':
        cluster='local'
    
    currwd = os.getcwd()
    print(currwd)
    #--------------------------------------------------------------------------
    # Ofiles folder
        
    if cluster=='haba':
        path_2_package="/rigel/theory/users/thn2112/ToroidalNetworks/sparse_weights/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='moto':
        path_2_package="/moto/theory/users/thn2112/ToroidalNetworks/sparse_weights/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='burg':
        path_2_package="/burg/theory/users/thn2112/ToroidalNetworks/sparse_weights/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='axon':
        path_2_package="/home/thn2112/ToroidalNetworks/sparse_weights/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='local':
        path_2_package="/Users/tuannguyen/ToroidalNetworks/sparse_weights/scripts"
        ofilesdir = path_2_package+"/Ofiles/"
        resultsdir = path_2_package + "/results/"



    if not os.path.exists(ofilesdir):
        os.makedirs(ofilesdir)
    
    
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    
    
    #--------------------------------------------------------------------------
    # The array of hashes
    c_Vec=np.arange(6)[[0,2,-1]]#12)
    # SoriE_mult_vec = (np.arange(4+1)/4)[2:]
    SoriE_mult = 1.0
    SoriI_mult = 1.0
    SoriF_mult = 1.0
    CVh_mult = 1.0
    L_mult_vec = (np.arange(-4,4+1)/4)
    CVL_mult = 1.0
    J_mult = 1.0#0.9
    beta_mult = 1.0#0.9
    gE_mult = 1.0#1/0.9
    gI_mult = 1.0
    hE_mult = 1.0#1.07/0.9
    hI_mult = 1.0
    
    for c in c_Vec:
        for L_mult in L_mult_vec:

            time.sleep(0.2)
            
            #--------------------------------------------------------------------------
            # Make SBTACH
            inpath = currwd + "/sim_multi_ori.py"
            c1 = "{:s} -c {:d} -Lm {:f}".format(
                inpath,c,L_mult)
            
            jobname="sim_multi_ori"+"-Lx{:.2f}-c-{:d}".format(
                L_mult,c)
            
            if not args2.test:
                jobnameDir=os.path.join(ofilesdir, jobname)
                text_file=open(jobnameDir, "w");
                os.system("chmod u+x "+ jobnameDir)
                text_file.write("#!/bin/sh \n")
                if cluster=='haba' or cluster=='moto' or cluster=='burg':
                    text_file.write("#SBATCH --account=theory \n")
                text_file.write("#SBATCH --job-name="+jobname+ "\n")
                text_file.write("#SBATCH -t 0-3:59  \n")
                text_file.write("#SBATCH --mem-per-cpu=10gb \n")
                text_file.write("#SBATCH --gres=gpu\n")
                text_file.write("#SBATCH -c 1 \n")
                text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
                text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
                text_file.write("python  -W ignore " + c1+" \n")
                text_file.write("echo $PATH  \n")
                text_file.write("exit 0  \n")
                text_file.close()

                if cluster=='axon':
                    os.system("sbatch -p burst " +jobnameDir);
                else:
                    os.system("sbatch " +jobnameDir);
            else:
                print (c1)



if __name__ == "__main__":
    runjobs()


