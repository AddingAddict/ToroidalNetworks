import os
import socket
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random
from tempfile import TemporaryDirectory


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
    K_Vec = [300]
    num_samp = 1#00
    
    outer_jobs = 1#0
    inner_jobs = 1#0

    with TemporaryDirectory() as temp_dir:
        
        for K in K_Vec:
            for outer_idx in range(outer_jobs):
                #--------------------------------------------------------------------------
                # Make SBTACH
                inpath = currwd + "/fit_relu.py"
                c1 = "{:s} -i $SLURM_ARRAY_TASK_ID -ns {:d} -K {:d}".format(
                    inpath,num_samp,K)
                jobname="{:s}-job_id={:d}-K-{:d}".format(
                    'fit_relu',outer_idx,K)

                time.sleep(0.5)
                
                if not args2.test:
                    jobnameDir=os.path.join(temp_dir, jobname)
                    text_file=open(jobnameDir, "w");
                    os.system("chmod u+x "+ jobnameDir)
                    text_file.write("#!/bin/sh \n")
                    if cluster=='haba' or cluster=='moto' or cluster=='burg':
                        text_file.write("#SBATCH --account=theory \n")
                    text_file.write("#SBATCH --job-name="+jobname+ "\n")
                    text_file.write("#SBATCH -t 0-11:59  \n")
                    text_file.write("#SBATCH --mem-per-cpu=10gb \n")
                    text_file.write("#SBATCH --gres=gpu\n")
                    text_file.write("#SBATCH -c 1 \n")
                    text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%A_%a.o # STDOUT \n")
                    text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%A_%a.e # STDERR \n")
                    text_file.write("python  -W ignore " + c1+" \n")
                    text_file.write("echo $PATH  \n")
                    text_file.write("exit 0  \n")
                    text_file.close()

                    if cluster=='axon':
                        os.system(f"sbatch -a {inner_jobs*outer_idx}-{inner_jobs*(outer_idx+1)-1} -p burst " +jobnameDir);
                    else:
                        os.system(f"sbatch -a {inner_jobs*outer_idx}-{inner_jobs*(outer_idx+1)-1} " +jobnameDir);
                else:
                    print (c1)



if __name__ == "__main__":
    runjobs()


