from models import GCNNet, GATNet, GATWithJK, TransfEnc


class Experiment(object):
    def __init__(self):
        pass

    def get_model_param_grid(self):
        return self.model_param_grid

    def get_trainer_param_grid(self):
        return self.trainer_param_grid

    def model_func(self):
        return self.model_func


class ExperimentGCN(Experiment):
    def __init__(self):
        super(ExperimentGCN, self).__init__()
        self.model_param_grid = {
            'dropout': [0.0],
            'hidden_dim': [32],
            'num_layers': [4]
        }
        self.trainer_param_grid = {
            'epochs': [200],
            'lr': [0.001],
            'batch_size': [8],
        }
        self.model_func = GCNNet

    def __str__(self):
        return 'GCNNet'


class ExperimentGAT(Experiment):
    def __init__(self):
        super(ExperimentGAT, self).__init__()
        self.model_param_grid = {
            'dropout': [0.0],
            'hidden_dim': [64],
            'heads': [2],
            'num_layers': [3],
        }

        self.trainer_param_grid = {
            'epochs': [200],
            'lr': [0.001],
            'batch_size': [8],
        }

        self.model_func = GATNet

    def __str__(self):
        return 'GATNet'


class ExperimentJK(Experiment):
    def __init__(self):
        super(ExperimentJK, self).__init__()
        self.model_param_grid = {
            'dropout': [0.3],
            'hidden_dim': [64],
            'heads': [1],
            'num_layers': [10]
        }

        self.trainer_param_grid = {
            'epochs': [200],
            'lr': [0.001],
            'batch_size': [1],
        }

        self.model_func = GATWithJK

    def __str__(self):
        return 'GATWithJK'


class ExperimentTransfEnc(Experiment):
    def __init__(self):
        super(ExperimentTransfEnc, self).__init__()
        self.model_param_grid = {
            'dropout': [0.3],
            'hidden_dim': [128],
            'heads': [1],
            'num_layers': [10]
        }

        self.trainer_param_grid = {
            'epochs': [200],
            'lr': [0.001],
            'batch_size': [1],

        }

        self.model_func = TransfEnc

    def __str__(self):
        return 'TransfEnc'
