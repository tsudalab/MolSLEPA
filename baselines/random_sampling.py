import sys
sys.path.append("..")
import numpy as np
from function.gap_prediction import GapPrediction
from assembler.assembler import Frag_assembler

def random_sampling(num_samples, vocab_size, vocab_len, itt_seed):
    x_current = []
    E_current = []
    for _ in range(num_samples):
        num_vocab = np.random.choice(vocab_size, size=vocab_len, replace=True)
        matches = np.zeros(vocab_size)
        for n in num_vocab:
            matches[n] += 1     
        smiles = assembler.junction(matches)  
        prop = predictor.predict(smiles)
        x_current.append(matches)
        E_current.append(prop)
    np.save('results/random_vocab30_K4/allseq_gap_'+str(num_samples)+'_'+str(itt_seed), x_current)
    np.save('results/random_vocab30_K4/allenergy_gap_'+str(num_samples)+'_'+str(itt_seed), E_current)
        
num_samples = 2000
vocab_size = 30
vocab_len = 4
model_dir = "/home/jiawen/molecular_design/molecule_generation/cli/zinc_vocab_1000_outputs/"
multisets = [
            'C1=CC=CC=C1', 'NC=O', 'C1=CSC=N1', 'C1=COC=C1', 'C1=CN=CN=C1', 
            'C1CCCCCC1', 'C1CNC1', 'CC(C)C', 'C1=NC=NO1', 'C1=CNN=N1', 
            'C1CCOCC1', 'C1CNCCN1', 'C1=NN=CS1', 'C1=CC2=NC=CN2C=C1','FCF', 
            'CNC(N)=O', 'C1=CCNCC1', 'C1=CSN=C1', 'C=CC=O', 'CCOC', 
            'OC(F)F', 'C1CC1', 'CCN(C)C', 'COC(C)=O', 'CCS', 
            'C1=CN=C2C=CC=CC2=C1', 'C=CC(N)=O', 'CC#N', 'C1C2CC3CC1CC(C2)C3', 'N[SH](=O)=O'
            ]
predictor = GapPrediction('min')
assembler = Frag_assembler(model_dir, multisets)

for itt_seed in range(5):
    random_sampling(num_samples, vocab_size, vocab_len, itt_seed)