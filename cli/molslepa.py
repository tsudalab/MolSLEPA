import sys
sys.path.append("..")
import numpy as np
from sklearn.preprocessing import StandardScaler
import physbo
import threading
import copy
from function.gap_prediction import GapPrediction
from assembler.assembler import Frag_assembler
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"



class ParallelThread(threading.Thread):
	def __init__(self, func, args, name=''):
		threading.Thread.__init__(self)
		self.name = name
		self.opt = func
		self.args = args
		self.result = self.opt(*self.args)

	def get_result(self):
		try:
			return self.result
		except Exception:
			return None

model_dir = "/home/jiawen/molecular_design/molecule_generation/cli/zinc_vocab_1000_outputs/"

class Sample():
    def __init__(self, multisets, function, opt):
        self.vocabulary = multisets
        self.items = np.arange(30)
        self.vocab_size = 30
        self.vocab_len = 4
        self.num_particle = 100
        self.seed = 0
        self.numiter = 20
        self.max_beta = 5
        self.opt = opt
        self.betas = np.linspace(0, self.max_beta, self.numiter)
        self.X_train = []
        self.X_train_scaled = []
        self.t_train = []

        self.MC_step = 50
        self.scaler = StandardScaler()
        self.predictor = function
        self.assembler = Frag_assembler(model_dir, self.vocabulary)
    
    def self_learning_population_annealing(self):
        
        print("Number of particle =", self.num_particle, self.max_beta, self.opt, self.seed, flush=True)
        #seed

        np.random.seed(self.seed)

        allenergy = np.zeros([self.numiter,self.num_particle])
        allenergy_surr = np.zeros([self.numiter,self.num_particle])
        allseq = [['']*self.num_particle for i in range(self.numiter)]


        #initial random state
        x_current = []
        E_current = []
        for _ in range(self.num_particle):
            num_vocab = np.random.choice(self.vocab_size, size=self.vocab_len, replace=True)
            matches = np.zeros(self.vocab_size)
            for n in num_vocab:
                matches[n] += 1     
            x_current.append(matches)
            smiles = self.assembler.junction(matches)  
            prop = self.predictor.predict(smiles)
            E_current.append(prop)
            
        x_current = np.array(x_current)
        E_current = np.array(E_current)
        

        for biter in range(self.numiter):
            print("iteration =",biter, flush=True)

            print("mean value of E =",np.mean(E_current),"+/-",np.std(E_current), flush=True)
            
            print("number of training data =",len(self.t_train), flush=True)

            if biter != 0:

                #resampling
                x_current, E_current_surr = self.resampling(self.betas[biter], self.betas[biter-1], E_current_surr, E_current_surr_prev, x_current)

                #MCMC
                threads = []
                for i in range(self.num_particle):
                    t = ParallelThread(self.MCMC, (x_current[i], E_current_surr[i], biter, gp), self.MCMC.__name__)
                    threads.append(t)
                for i in range(self.num_particle):
                    threads[i].start()
                for i in range(self.num_particle):
                    threads[i].join()
                for i in range(self.num_particle):
                    x_current[i], E_current_surr[i] = threads[i].get_result()
                for i in range(self.num_particle):
                    E_current[i] = self.predictor.predict(self.assembler.junction(x_current[i]))

                E_current_surr_prev = copy.deepcopy(E_current_surr)

            #add training data
            for i in range(len(x_current)):
                self.X_train.append(x_current[i])
                self.t_train.append(E_current[i])


            #training GP
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            

            cov = physbo.gp.cov.gauss( np.array(self.X_train_scaled).shape[1],ard = False )
            mean = physbo.gp.mean.const()
            lik = physbo.gp.lik.gauss()
            gp = physbo.gp.model(lik=lik,mean=mean,cov=cov)
            config = physbo.misc.set_config()
            gp.fit(np.array(self.X_train_scaled), np.array(self.t_train), config)
            gp.prepare(np.array(self.X_train_scaled), np.array(self.t_train))


            #surrogate E
            X_test = []

            for i in range(len(x_current)):
                X_test.append(x_current[i])

            # print(np.shape(X_test))
            X_test_scaled = self.scaler.transform(X_test)
            E_current_surr = gp.get_post_fmean(np.array(self.X_train_scaled), np.array(X_test_scaled))
            print('model update!')

            if biter == 0: E_current_surr_prev = copy.deepcopy(E_current_surr)

            #result for each beta

            allenergy[biter] = E_current
            allenergy_surr[biter] = E_current_surr_prev
            allseq[biter] = x_current

        allseq_all = copy.deepcopy(allseq)
        allenergy_all = copy.deepcopy(allenergy)

        # resampling

        for biter in range(len(allenergy)):

            prob = np.exp(-self.betas[biter]*allenergy[biter] + self.betas[biter]*allenergy_surr[biter])
            prob = prob/np.sum(prob)
            id = np.random.choice(self.num_particle,self.num_particle,p=prob, replace=True)
            allseq[biter] = allseq[biter][id]
            allenergy[biter] = allenergy[biter][id]
       
        np.save('results/molslepa_vocab30_K4/allseq_gap_'+self.opt+'_T'+str(self.max_beta)+'_s_'+str(self.seed)+'_nump_'+str(self.num_particle)+'_iter'+str(self.numiter), allseq)
        np.save('results/molslepa_vocab30_K4/allenergy_gap_'+self.opt+'_T'+str(self.max_beta)+'_s_'+str(self.seed)+'_nump_'+str(self.num_particle)+'_iter'+str(self.numiter), allenergy)
    

    def MCMC(self, x_curr, E_curr, biter, gp):
        for _ in range(self.MC_step):
            x_proposal = copy.deepcopy(x_curr)
            num_vocab = np.nonzero(x_proposal)
            
            x_proposal[np.random.choice(num_vocab[0], 1)] -= 1
            x_proposal[np.random.randint(0, len(x_proposal), 1)] += 1  
            X_test_scaled = self.scaler.transform(x_proposal.reshape(1,-1))

            E_proposal = gp.get_post_fmean(np.array(self.X_train_scaled), np.array(X_test_scaled))[0]

            if np.exp(self.betas[biter]*(E_curr-E_proposal)) > np.random.rand():
                x_curr = x_proposal
                E_curr = E_proposal
        return x_curr, E_curr

    def resampling(self, beta_current, beta_prev, E_current, E_current_prev, x_current):
        prob = np.exp(-beta_current*E_current + beta_prev*E_current_prev)
        prob = prob/np.sum(prob)
        ids = np.random.choice(self.num_particle,self.num_particle, p=prob, replace=True)
        x_current = np.array(x_current)[ids]
        E_current = np.array(E_current)[ids]
        return x_current, E_current 
    
class DosEstimation():
    def __init__(self):
        self.num_discretize = 50
        
    def multi_histogram(self, betas, allseq, allenergy, vocab_size, thres):
        numiter = len(betas)
        estdists = np.zeros([numiter, self.num_discretize])
        # make histogram
        E_max = np.max(allenergy)+0.000001
        E_min = np.min(allenergy)-0.000001
        for biter in range(numiter):
            for e in allenergy[biter]:
                index_current = int((e-E_min)/(E_max-E_min)*self.num_discretize)
                estdists[biter][index_current] += 1        
            
        #multiple histogram
        estsum = np.sum(estdists,axis = 0)
        es = np.zeros(self.num_discretize)
        width = E_max-E_min

        for i in range(self.num_discretize):
            low = E_min + i*width/self.num_discretize
            high = E_min + (i+1)*width/self.num_discretize
            es[i] = (high+low)/2

        f = np.zeros(numiter)
        for fiter in range(10):
            estn = np.zeros(self.num_discretize)
            for i in range(self.num_discretize):
                res = 0
                for j in range(numiter):
                    res = res + sum(estdists[j,:])*np.exp(-betas[j]*es[i]+f[j])
                estn[i] = estsum[i]/res

            for j in range(numiter):
                f[j] = -np.log(np.dot(estn, np.exp(-betas[j]*es)))

        estn = estn/np.sum(estn)

        R = np.zeros([self.num_discretize,numiter])
        for i in range(self.num_discretize):    
            for j in range(numiter):
                R[i][j] = np.exp(-betas[j]*es[i] + f[j])
            R[i] = R[i]/np.sum(R[i])

        #observable
        counter = np.zeros(vocab_size)
        for biter in range(numiter):
            matches= [allseq[biter][i] for i,x in enumerate(allenergy[biter]) if x <= thres]
            matchenergy = [allenergy[biter][i] for i,x in enumerate(allenergy[biter]) if x <= thres]       
            for i in range(len(matches)):
                num_vocab = np.nonzero(matches[i])[0]
                for j in num_vocab:  
                    for _ in range(int(matches[i][j])):  
                        ik = int((matchenergy[i]-E_min)/(E_max-E_min)*self.num_discretize)
                        counter[j] += R[ik][biter]*np.exp(betas[biter]*es[ik]-f[biter])
        if (np.sum(counter) > 0):
            counter /= np.sum(counter)
        return counter
    

def main():
    multisets = [
            'C1=CC=CC=C1', 'NC=O', 'C1=CSC=N1', 'C1=COC=C1', 'C1=CN=CN=C1', 
            'C1CCCCCC1', 'C1CNC1', 'CC(C)C', 'C1=NC=NO1', 'C1=CNN=N1', 
            'C1CCOCC1', 'C1CNCCN1', 'C1=NN=CS1', 'C1=CC2=NC=CN2C=C1','FCF', 
            'CNC(N)=O', 'C1=CCNCC1', 'C1=CSN=C1', 'C=CC=O', 'CCOC', 
            'OC(F)F', 'C1CC1', 'CCN(C)C', 'COC(C)=O', 'CCS', 
            'C1=CN=C2C=CC=CC2=C1', 'C=CC(N)=O', 'CC#N', 'C1C2CC3CC1CC(C2)C3', 'N[SH](=O)=O'
            ]
    opt = 'max' # or 'min'
    function = GapPrediction(opt)
    slepa = Sample(multisets, function, opt)
    slepa.self_learning_population_annealing()

main()
