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
    parser.add_argument('--njob', '-nj',  help='which number job', type=int, default=133)
    parser.add_argument('--nrep', '-nr',  help='which number repetition', type=int, default=0)
    
    args2 = parser.parse_args()
    args = vars(args2)
    njob= args['njob']
    nrep= args['nrep']
    
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
    for script in ('sim_ring_perturbed','sim_ring_perturbed_large'):
        if cluster=='axon':
            ntry_skip = 10
            ntry_Vec=range(0,100,ntry_skip)
            
            for ntry in ntry_Vec:
                inpath = currwd + "/{:s}.py".format(script)
                c1 = "{:s} -nj {:d} -nr {:d} -nt $SLURM_ARRAY_TASK_ID".format(inpath,njob,nrep)
                
                jobname="{:s}".format(script)+"-njob-{:d}-nrep-{:d}".format(njob,nrep)
                
                if not args2.test:
                    jobnameDir=os.path.join(ofilesdir, jobname)
                    text_file=open(jobnameDir, "w");
                    os.system("chmod u+x "+ jobnameDir)
                    text_file.write("#!/bin/sh \n")
                    if cluster=='haba' or cluster=='moto' or cluster=='burg':
                        text_file.write("#SBATCH --account=theory \n")
                    text_file.write("#SBATCH --job-name="+jobname+ "\n")
                    text_file.write("#SBATCH --array={:d}-{:d}\n".format(ntry,ntry+ntry_skip-1))
                    text_file.write("#SBATCH -t 0-11:59  \n")
                    text_file.write("#SBATCH --mem-per-cpu=8gb \n")
                    text_file.write("#SBATCH --gres=gpu\n")
                    text_file.write("#SBATCH -c 1 \n")
                    text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%A_%a.o # STDOUT \n")
                    text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%A_%a.e # STDERR \n")
                    text_file.write("python  -W ignore " + c1+" \n")
                    text_file.write("echo $PATH  \n")
                    text_file.write("exit 0  \n")
                    text_file.close()
                    
                    os.system("sbatch " +jobnameDir);
                else:
                    print (c1)
        else:
            # The array of hashes
            ntry_Vec=range(100)
            
            for ntry in ntry_Vec:

                time.sleep(0.2)
                
                #--------------------------------------------------------------------------
                # Make SBTACH
                inpath = currwd + "/sim_ring_perturbed.py"
                c1 = "{:s} -nj {:d} -nr {:d} -nt {:d}".format(inpath,njob,nrep,ntry)
                
                jobname="sim_ring_perturbed"+"-njob-{:d}-nrep-{:d}-ntry-{:d}".format(njob,nrep,ntry)
                
                if not args2.test:
                    jobnameDir=os.path.join(ofilesdir, jobname)
                    text_file=open(jobnameDir, "w");
                    os.system("chmod u+x "+ jobnameDir)
                    text_file.write("#!/bin/sh \n")
                    if cluster=='haba' or cluster=='moto' or cluster=='burg':
                        text_file.write("#SBATCH --account=theory \n")
                    text_file.write("#SBATCH --job-name="+jobname+ "\n")
                    text_file.write("#SBATCH -t 0-11:59  \n")
                    text_file.write("#SBATCH --mem-per-cpu=8gb \n")
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


