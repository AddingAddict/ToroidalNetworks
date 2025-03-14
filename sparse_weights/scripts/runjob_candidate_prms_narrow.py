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
    with open('./../notebooks/candidate_prms.txt', 'r') as f:
        id_list = f.read().split('\n')
        max_njobs = len(id_list)
    print('total number of jobs: {:d}'.format(max_njobs))

    if cluster=='axon':
        njob_skip = 20
        njob_Vec=range(0,max_njobs,njob_skip)
            
        for njob in njob_Vec:
            time.sleep(0.2)
            
            #--------------------------------------------------------------------------
            # Make SBTACH
            inpath = currwd + "/refit_candidate_prms_narrow.py"
            c1 = "{:s} -n $SLURM_ARRAY_TASK_ID".format(inpath)
            
            jobname="refit_candidate_prms_narrow"
            
            if not args2.test:
                jobnameDir=os.path.join(ofilesdir, jobname)
                text_file=open(jobnameDir, "w");
                os.system("chmod u+x "+ jobnameDir)
                text_file.write("#!/bin/sh \n")
                if cluster=='haba' or cluster=='moto' or cluster=='burg':
                    text_file.write("#SBATCH --account=theory \n")
                text_file.write("#SBATCH --job-name="+jobname+ "\n")
                text_file.write("#SBATCH --array={:d}-{:d}\n".format(njob,njob+njob_skip-1))
                text_file.write("#SBATCH -t 0-11:59  \n")
                text_file.write("#SBATCH --mem-per-cpu=15gb \n")
                text_file.write("#SBATCH --gres=gpu\n")
                text_file.write("#SBATCH -c 1 \n")
                text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
                text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
                text_file.write("python  -W ignore " + c1+" \n")
                text_file.write("echo $PATH  \n")
                text_file.write("exit 0  \n")
                text_file.close()

                os.system("sbatch " +jobnameDir);
            else:
                print (c1)
    else:
        # The array of hashes
        njob_Vec=range(max_njobs)
        
        for njob in njob_Vec:

            time.sleep(0.2)
            
            #--------------------------------------------------------------------------
            # Make SBTACH
            inpath = currwd + "/refit_candidate_prms_narrow.py"
            c1 = "{:s} -n {:d}".format(inpath,njob)
            
            jobname="refit_candidate_prms_narrow"
            
            if not args2.test:
                jobnameDir=os.path.join(ofilesdir, jobname)
                text_file=open(jobnameDir, "w");
                os.system("chmod u+x "+ jobnameDir)
                text_file.write("#!/bin/sh \n")
                if cluster=='haba' or cluster=='moto' or cluster=='burg':
                    text_file.write("#SBATCH --account=theory \n")
                text_file.write("#SBATCH --job-name="+jobname+ "\n")
                text_file.write("#SBATCH -t 0-11:59  \n")
                text_file.write("#SBATCH --mem-per-cpu=15gb \n")
                text_file.write("#SBATCH --gres=gpu\n")
                text_file.write("#SBATCH -c 1 \n")
                text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
                text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
                text_file.write("python  -W ignore " + c1+" \n")
                text_file.write("echo $PATH  \n")
                text_file.write("exit 0  \n")
                text_file.close()
                
                os.system("sbatch " +jobnameDir);
            else:
                print (c1)



if __name__ == "__main__":
    runjobs()


