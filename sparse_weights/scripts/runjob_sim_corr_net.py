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
    c_Vec=np.arange(7)[[0,1,3,5,6]]#12)
    # SoriE_mult_vec = (np.arange(4+1)/4)[2:]
    SoriE_mult_vec = np.array([1.0,])
    SoriI_mult_vec = np.array([1.0,])
    SoriF_mult_vec = np.array([1.0,])
    CVh_mult_vec = np.array([1.0,])
    L_mult_vec = np.array([1.0,])
    CVL_mult_vec = np.array([1.0,])
    J_mult = 1.0
    beta_mult = 1.0
    gE_mult = 1.0
    gI_mult = 1.0
    hE_mult = 1.0
    hI_mult = 1.0
    
    for c in c_Vec:
        for SoriE_mult in SoriE_mult_vec:
            for SoriI_mult in SoriI_mult_vec:
                for SoriF_mult in SoriF_mult_vec:
                    for CVh_mult in CVh_mult_vec:
                        for L_mult in L_mult_vec:
                            for CVL_mult in CVL_mult_vec:

                                time.sleep(0.2)
                                
                                #--------------------------------------------------------------------------
                                # Make SBTACH
                                inpath = currwd + "/sim_corr_net.py"
                                c1 = "{:s} -c {:d} -SoriEm {:f} -SoriIm {:f} -SoriFm {:f} -CVhm {:f} -Jm {:f} -betam {:f} -gEm {:f} -gIm {:f} -hEm {:f} -hIm {:f} -Lm {:f} -CVLm {:f}".format(
                                    inpath,c,SoriE_mult,SoriI_mult,SoriF_mult,CVh_mult,J_mult,beta_mult,gE_mult,gI_mult,hE_mult,hI_mult,L_mult,CVL_mult)
                                
                                if np.isclose(SoriE_mult,1.0) and np.isclose(SoriI_mult,1.0) and\
                                    np.isclose(SoriF_mult,1.0) and np.isclose(CVh_mult,1.0) and\
                                    np.isclose(J_mult,1.0) and np.isclose(beta_mult,1.0) and\
                                    np.isclose(gE_mult,1.0) and np.isclose(gI_mult,1.0) and\
                                    np.isclose(hE_mult,1.0) and np.isclose(hI_mult,1.0) and\
                                    np.isclose(L_mult,1.0) and np.isclose(CVL_mult,1.0):
                                    jobname="sim_corr_net"+"-c-{:d}".format(c)
                                else:
                                    jobname="sim_corr_net"+"-SoriEx{:.2f}-SoriIx{:.2f}-SoriFx{:.2f}-CVhx{:.2f}-Jx{:.2f}-betax{:.2f}-gEx{:.2f}-gIx{:.2f}-hEx{:.2f}-hIx{:.2f}-Lx{:.2f}-CVLx{:.2f}-c-{:d}".format(
                                        SoriE_mult,SoriI_mult,SoriF_mult,CVh_mult,J_mult,beta_mult,gE_mult,gI_mult,hE_mult,hI_mult,L_mult,CVL_mult,c)
                                
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


