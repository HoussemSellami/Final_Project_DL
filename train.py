import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed

from experiments import *
from models import *
from utils import *

MAX_PATIENCE = 3


class Trainer(object):
    def __init__(self):
        self.target_func = lambda x: x

    def train(self, model, lr, loader, epochs):
        self.patience = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-8)

        best_val_epoch = 0
        best_val_performance = 0
        best_model_params = None

        train_format = '{:.4f}'
        val_format = '{:.4f}'

        self.target_func = lambda x: x

        train = []
        val = []
        scores_train = []
        scores_val = []

        for epoch in range(1, epochs):
            all_train_labels = []
            all_train_preds = []
            all_val_labels = []
            all_val_preds = []

            model.train()
            for k, data in enumerate(loader):
                optimizer.zero_grad()
                if type(model) == TransfEnc:
                    out, scores_t = model(data)
                else:
                    out = model(data)
                out = torch.squeeze(out)

                mask = data.train_mask
                # no labels in this batch -> skip
                if mask.sum() == 0:
                    continue
                loss = F.nll_loss(out[mask], torch.squeeze(data.y[mask]))

                train_y = self.target_func(out[mask].max(1)[1])
                train_labels = self.target_func(torch.squeeze(data.y[mask]))

                all_train_labels.append(train_y.detach().numpy())
                all_train_preds.append(train_labels.numpy())

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for val_data in loader:
                    if type(model) == TransfEnc:
                        out, scores_v = model(val_data)
                    else:
                        out = model(val_data)
                    out = torch.squeeze(out)

                    # validation
                    mask = val_data.val_mask

                    val_y = self.target_func(out[mask].max(1)[1])
                    val_labels = self.target_func(torch.squeeze(val_data.y[mask]))

                    all_val_preds.append(val_y)
                    all_val_labels.append(val_labels)

            acc_val, acc_train = evaluate(all_val_labels, all_val_preds, all_train_labels, all_train_preds)

            train.append(acc_train)
            val.append(acc_val)
            if type(model) == TransfEnc:
                scores_train.append(torch.mean(scores_t, axis=2))
                scores_val.append(torch.mean(scores_v, axis=2))

            if val[-1] > best_val_performance:
                best_val_epoch = epoch - 1  # epoch starts from 1 on
                best_val_performance = val[-1]
                best_model_params = model.state_dict()

            if len(val) > 1:
                # is the sum of all (enabled) tasks
                if np.nansum(val[-1]) <= np.nansum(val[-2]):
                    self.patience += 1
                else:
                    self.patience = 0

            if epoch % 1 == 0:
                output_string = "Epoch {} - " + "Train: " + train_format + " - Val: " + val_format
                print(output_string.format(epoch, train[-1], val[-1]))

            if self.patience == MAX_PATIENCE:
                print("Early stopping on validation loss")
                return [train, val], best_val_epoch, best_model_params, [scores_train, scores_val]

        return [train,  val], best_val_epoch, best_model_params, [scores_train, scores_val]

    def test(self, model, loader):
        for k, data in enumerate(loader):
            with torch.no_grad():
                model.eval()
                if type(model) == TransfEnc:
                    out, scores_v = model(data)
                else:
                    out = model(data)

                out = torch.squeeze(out)

                mask = data.test_mask
                if mask.sum() == 0:
                    continue

                test_y = self.target_func(out[mask].max(1)[1])
                test_labels = self.target_func(torch.squeeze(data.y[mask]))
        test_acc = accuracy_score(test_labels, test_y)
        cm = conf_matrix_test(test_labels, test_y)
        df_cm = pd.DataFrame(cm, range(1, 7), range(1, 7))
        fig = plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        plt.show()
        print("----")
        print("Test: ", test_acc)
        # plt.savefig('cm.png', dpi=fig.dpi)
        return test_acc

    def run_experiments(self, root_dir, experimenter):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = 'CiteSeer'
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
        dataset = Planetoid(path, dataset, "full")
        processed_dir = os.path.join(root_dir, 'processed')

        model_params = experimenter.get_model_param_grid()
        trainer_params = experimenter.get_trainer_param_grid()
        configs_model = ParameterGrid(model_params)
        configs_trainer = ParameterGrid(trainer_params)
        results = []
        best_model = None
        best_acc = 0
        i = 0
        for model_config in configs_model:
            for trainer_config in configs_trainer:
                i = i + 1
                print('Training model {}: {:d} of {:d} '.format(experimenter.model_func, i,
                                                                len(configs_model) * len(configs_trainer)))
                # re-setting the random number generator
                random_seed = 24
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)

                # Split data
                batch_size = trainer_config['batch_size']
                loader = DataLoader(dataset, batch_size=batch_size)

                num_features = [data.x.numpy().shape[1] for data in loader][0]

                model = experimenter.model_func(num_outputs=6, input_dim=num_features, **model_config).to(device)
                # model, device, lr, loader, epochs
                accs, best_epoch, model_params, scores = self.train(model,
                                                                    lr=trainer_config['lr'],
                                                                    loader=loader,
                                                                    epochs=trainer_config['epochs'])
                train_acc, val_acc = accs

                if val_acc[best_epoch] > best_acc:
                    best_acc = val_acc[best_epoch]
                    best_model = model.load_state_dict(model_params)
                    # save the currently best model to disk
                    with open(os.path.join(processed_dir, 'best_model_' + str(experimenter) + '.pt'), 'wb') as fp:
                        torch.save(model_params, fp, pickle_module=pickle,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL)
                    # merge both param dicts
                    best_params = {}
                    best_params.update(model_config)
                    best_params.update(trainer_config)

                test_acc = self.test(model, loader)
                # merge both param dicts
                all_params_dict = {}
                all_params_dict.update(model_config)
                all_params_dict.update(trainer_config)
                all_params_dict['val_acc'] = val_acc[best_epoch]
                all_params_dict['train_acc'] = train_acc[best_epoch]
                all_params_dict['test_acc'] = test_acc
                results.append(all_params_dict)
                pd.DataFrame(results).sort_values('val_acc').to_csv(os.path.join(processed_dir, 'results_' + str(experimenter) + '.csv'))

        return best_model


if __name__ == '__main__':
    # You can choose which networks you want to train by adding or removing elements from the experimenters list
    experimenters = [ExperimentGCN(),
                     ExperimentGAT(),
                     ExperimentJK(),
                     ExperimentTransfEnc()]

    def dispatch_experiment(experimenter):
        trainer = Trainer()
        trainer.run_experiments('./data/CiteSeer',
                                experimenter=experimenter)

    Parallel(n_jobs=1)(delayed(dispatch_experiment)(experiment) for experiment in experimenters)
