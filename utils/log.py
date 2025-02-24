import os
import wandb
import pickle
from dataclasses import dataclass, field, asdict

@dataclass
class IterLog():

    experiment_name: str

    running_train_loss_raw: float = 0
    running_train_loss_indices: float = 0

    running_val_loss_raw: float = 0
    running_val_loss_indices: float = 0

    running_train_metric_raw: float = 0
    running_train_metric_indices: float = 0

    running_val_metric_raw: float = 0
    running_val_metric_indices: float = 0

    total_train: int = 0
    total_val: int = 0

    

    def on_iter_start(self) -> None:
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), (int, float)):
                setattr(self, attr, 0.0)

    def add_on_train_iter_end(self, train_loss_raw, train_loss_indiced, train_metric_raw, train_metric_indiced) -> None:

        self.running_train_loss_raw += train_loss_raw
        self.running_train_loss_indices += train_loss_indiced
        
        self.running_train_metric_raw += train_metric_raw
        self.running_train_metric_indices += train_metric_indiced

        self.total_train += 1

    def add_on_val_iter_end(self, val_loss_raw, val_loss_indiced, val_metric_raw, val_metric_indiced) -> None:

        self.running_val_loss_raw += val_loss_raw
        self.running_val_loss_indices += val_loss_indiced
        
        self.running_val_metric_raw += val_metric_raw
        self.running_val_metric_indices += val_metric_indiced

        self.total_val += 1

    def on_epoch_end(self, ep) -> None:
        to_log = {}
        for attr in self.__dict__:
            
            print(attr)

            val = getattr(self, attr)
            
            if isinstance(val, str) or 'total' in attr:
                continue

            if 'train' in attr:
                setattr(self, attr, val / self.total_train)
            else:
                setattr(self, attr, val / self.total_val)            

            to_log[('/'.join([getattr(self, 'experiment_name'), attr]))] = getattr(self, attr)

        to_log['epoch'] = ep

        wandb.log(to_log)

@dataclass
class TrainLog():

    experiment_name: str

    train_loss_raw: list[float] = field(default_factory=list)
    train_loss_indices: list[float] = field(default_factory=list)

    val_loss_raw: list[float] = field(default_factory=list)
    val_loss_indices: list[float] = field(default_factory=list)

    train_metric_raw: list[float] = field(default_factory=list)
    train_metric_indices: list[float] = field(default_factory=list)

    val_metric_raw: list[float] = field(default_factory=list)
    val_metric_indices: list[float] = field(default_factory=list)

    def on_epoch_end(self, iter_log: IterLog):
        for k, v in iter_log.__dict__.items():
            if 'running' in k:
                k = k.lstrip('running_')
                lst = getattr(self, k)
                lst.append(v)

    def save_pickle(self):
        
        if not os.path.isdir('/home/al/projects/global_rs_project/results'):
            os.mkdir('/home/al/projects/global_rs_project/results')        

        with open(f'/home/al/projects/global_rs_project/results/{self.experiment_name}.pkl', 'wb') as ouf:
             pickle.dump(asdict(self), ouf)