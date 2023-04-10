import sys
sys.path.append("..")
import numpy as np
import copy
from function.gap_prediction import GapPrediction
from assembler.assembler import Frag_assembler

vocab_size = 30
vocab_len = 4
num_particle = 100
num_seed = 5
model_dir = "/home/jiawen/molecular_design/molecule_generation/cli/zinc_vocab_1000_outputs/"
multisets = [
            'C1=CC=CC=C1', 'NC=O', 'C1=CSC=N1', 'C1=COC=C1', 'C1=CN=CN=C1', 
            'C1CCCCCC1', 'C1CNC1', 'CC(C)C', 'C1=NC=NO1', 'C1=CNN=N1', 
            'C1CCOCC1', 'C1CNCCN1', 'C1=NN=CS1', 'C1=CC2=NC=CN2C=C1','FCF', 
            'CNC(N)=O', 'C1=CCNCC1', 'C1=CSN=C1', 'C=CC=O', 'CCOC', 
            'OC(F)F', 'C1CC1', 'CCN(C)C', 'COC(C)=O', 'CCS', 
            'C1=CN=C2C=CC=CC2=C1', 'C=CC(N)=O', 'CC#N', 'C1C2CC3CC1CC(C2)C3', 'N[SH](=O)=O'
            ]
######################
# Parameters
######################

numiter = 20
num_discretize = 50
opt = 'max'

predictor = GapPrediction(opt)
assembler = Frag_assembler(model_dir, multisets)


for itt_seed in range(num_seed):

    print('')
    print(itt_seed , "th independent runs")
    print("Number of particle =", num_particle)

    #seed
    np.random.seed(itt_seed)


    allenergy = np.zeros([numiter,num_particle])
    allseq = [['']*num_particle for i in range(numiter)]
    estdists = np.zeros([numiter,num_discretize])

    #Initial state
    x_current = []
    E_current = []
    for _ in range(num_particle):
        num_vocab = np.random.choice(vocab_size, size=vocab_len, replace=True)
        matches = np.zeros(vocab_size)
        for n in num_vocab:
            matches[n] += 1  
        x_current.append(matches)
        smiles = assembler.junction(matches)  
        prop = predictor.predict(smiles)
        E_current.append(prop)
    x_current = np.array(x_current)
    E_current = np.array(E_current)


    for biter in range(0,numiter):
        print("iteration =",biter)

        print("mean value of E =",np.mean(E_current),"+/-",np.std(E_current))


        top = np.array(E_current).argsort()

        select_smi = []
        select_x = []
        select_E = []

        for i in range(int(0.1*num_particle)):
            for j in range(len(top)):
                select_x.append(x_current[top[j]])
                select_E.append(E_current[top[j]])
                break

        generate_pep = []
        generate_smi = []
        generate_E = []

        k = 0

        for i in range(len(select_x)):

            for j in range(int(num_particle/len(select_x))-1):
                
                x_proposal = copy.deepcopy(x_current[i])
                num_vocab = np.nonzero(x_proposal)
                x_proposal[np.random.choice(num_vocab[0], 1)] -= 1
                x_proposal[np.random.randint(0, len(x_proposal), 1)] += 1  
                smiles = assembler.junction(x_proposal)  
                E_proposal = predictor.predict(smiles)
                k+=1

                generate_pep.append(x_proposal)
                generate_E.append(E_proposal)


        x_current = np.concatenate([np.array(select_x), np.array(generate_pep)], 0)
        E_current = np.concatenate([np.array(select_E), np.array(generate_E)], 0)

        allenergy[biter] = E_current
        allseq[biter] = x_current

    print(allenergy)
    np.save('results/es_vocab30_K4/allseq_gap_max_'+str(num_particle)+'_'+str(itt_seed), allseq)
    np.save('results/es_vocab30_K4/allenergy_gap_max_'+str(num_particle)+'_'+str(itt_seed), allenergy)