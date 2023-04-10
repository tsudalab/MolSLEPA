from assembler.wrapper import VaeWrapper
import random
import numpy as np



class Frag_assembler():
    def __init__(self, model_dir, vocabulary):
        with VaeWrapper(model_dir) as model:
            self.model = model
        self.vocabulary = vocabulary
        self.number_pfa = 10

    def junction(self, matches):
        num_vocab = np.nonzero(matches)[0]
        for n in num_vocab:
            for _ in range(matches[n]-1):
             num_vocab = np.append(num_vocab, n)  
        prob_smiles = []
        for _ in range(self.number_pfa):
            random.shuffle(num_vocab)
            init_smiles = [self.vocabulary[num_vocab[0]]]
            for n in num_vocab[1:]:
                embeddings = self.model.encode(self.vocabulary[n])
                junction_smiles =self. model.decode(embeddings, self.vocabulary[n], scaffolds=init_smiles)
                init_smiles = junction_smiles
            prob_smiles.append([init_smiles[0]])
        return prob_smiles
        
    def encode(self, smiles):
        embeddings = self.model.encode(smiles)
        return embeddings[0]