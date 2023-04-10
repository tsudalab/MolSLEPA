import chemprop
from collections import defaultdict
import numpy as np

class GapPrediction():
    def __init__(self, opt):
        arguments = [
        '--test_path', '/dev/null',
        '--preds_path', '/dev/null',
        '--checkpoint_dir', '../function/2.6m_checkpoints']
        self.args = chemprop.args.PredictArgs().parse_args(arguments)
        self.model_objects = chemprop.train.load_model(args=self.args)
        self.opt = opt
    
    def predict(self, smiles):
        try:
            preds = chemprop.train.make_predictions(args=self.args, smiles=smiles, model_objects=self.model_objects)
            if self.opt == 'min':
                return np.mean(preds)
            else:
                return -np.mean(preds)
        except:
            # invalid smiles
            return 0



