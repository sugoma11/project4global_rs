import sys
import torch
import wandb
import hydra
import datetime
import itertools
from torch import optim
from models import get_model
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils.log import IterLog, TrainLog
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import StepLR

from utils.metrics import accuracy
from utils.data import load_pickled_ds, NumpyDataset



def train_and_evaluate(experiment_name: str, model_name: str, train_loader: DataLoader, val_loader: DataLoader,
                       lr: float, weight_decay: float, loss_fn, metric_fn,
                       device: str, num_epochs: int, step2decay: int, decay_lr: int, seed: int):
    
    
    indiced_sample, raw_sample = next(iter(train_loader))[:2]
    n_bands_raw, n_bands_indices = indiced_sample.shape[1], raw_sample.shape[1]
    input_size = indiced_sample.shape[2]

    model_class = get_model(model_name)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    model_indiced = model_class(n_classes=15, n_bands=n_bands_indices, input_size=input_size)
    model_raw = model_class(n_classes=15, n_bands=n_bands_raw, input_size=input_size)

    model_indiced.to(device)
    model_raw.to(device)

    loss_fn_raw = loss_fn().to(device)
    loss_fn_indiced = loss_fn().to(device)


    optimizer_raw = optim.AdamW(model_raw.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_indiced = optim.AdamW(model_indiced.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_raw = StepLR(optimizer_raw, step_size=step2decay, gamma=decay_lr)
    scheduler_indiced = StepLR(optimizer_indiced, step_size=step2decay, gamma=decay_lr)


    iter_log = IterLog(experiment_name)
    train_log = TrainLog(experiment_name)

    for epoch in range(num_epochs):
        
        iter_log.on_iter_start()

        model_indiced.train() 
        model_raw.train() 


        for inputs_raw, inputs_indiced, labels in train_loader:
            inputs_raw, inputs_indiced, labels = inputs_raw.to(device, dtype=torch.float), inputs_indiced.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

            optimizer_raw.zero_grad()
            optimizer_indiced.zero_grad()

            outputs_raw = model_raw.forward(inputs_raw)
            outputs_indiced = model_indiced.forward(inputs_indiced)

            loss_raw = loss_fn_raw(outputs_raw, labels)
            loss_raw.backward()

            loss_indiced = loss_fn_indiced(outputs_indiced, labels)
            loss_indiced.backward()

            optimizer_raw.step()
            optimizer_indiced.step()

            scheduler_raw.step()
            scheduler_indiced.step()

            _, predicted_raw = torch.max(outputs_raw, 1)
            _, predicted_indiced = torch.max(outputs_indiced, 1)

            metric_raw = metric_fn(predicted_raw, labels)
            metric_indiced = metric_fn(predicted_indiced, labels)

            iter_log.add_on_train_iter_end(train_loss_indiced=loss_indiced.item(), train_loss_raw=loss_raw.item(),
                                           train_metric_raw=metric_raw.item(), train_metric_indiced=metric_indiced.item())

        model_raw.eval()
        model_indiced.eval()

        with torch.no_grad(): 
            for inputs_raw, inputs_indiced, labels in val_loader:
                inputs_raw, inputs_indiced, labels = inputs_raw.to(device, dtype=torch.float), inputs_indiced.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

                outputs_raw = model_raw.forward(inputs_raw)
                outputs_indiced = model_indiced.forward(inputs_indiced)

                loss_raw = loss_fn_raw(outputs_raw, labels)
                loss_indiced = loss_fn_indiced(outputs_indiced, labels)

                _, predicted_raw = torch.max(outputs_raw, 1)
                _, predicted_indiced = torch.max(outputs_indiced, 1)
                
                metric_raw = metric_fn(predicted_raw, labels)
                metric_indiced = metric_fn(predicted_indiced, labels)

                iter_log.add_on_val_iter_end(val_loss_indiced=loss_indiced.item(), val_loss_raw=loss_raw.item(),
                                             val_metric_raw=metric_raw.item(), val_metric_indiced=metric_indiced.item())

        # wandb loggining inside
        iter_log.on_epoch_end(epoch)
        print(f"Epoch [{epoch}/{num_epochs}], {iter_log}")

        train_log.on_epoch_end(iter_log)
    
    train_log.save_pickle()

# @hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    num_epochs = cfg.num_epochs
    project_name = cfg.project_name

    run = wandb.init(
        project=project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"benchmark_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        reinit=True  
    )

    experiment_configs = list(itertools.product(cfg.model_names, 
                                                cfg.sentinel_number,
                                                cfg.band_order, cfg.requested_indices, 
                                                cfg.batch_sizes, cfg.learning_rates, cfg.decay_lr,
                                                cfg.step2decay, cfg.weight_decay, 
                                                cfg.seeds,
                                                cfg.device))
    
    for model_name, sentinel_number, band_order, requested_indices, bs, lr, decay_lr, step2decay, weight_decay, seed, device in experiment_configs:

        
        experiment_name = str({
        "Model": model_name,
        "BS": bs,
        "LR": lr,
        "Indixes": ';'.join(requested_indices),
        "Sentinel": sentinel_number,
        "seed": seed
        }).replace("'", '')

        print(f"Running experiment: {experiment_name}")

        print(( model_name, sentinel_number, band_order, requested_indices, bs, lr, decay_lr, step2decay, weight_decay, seed, device))

        if sentinel_number is 1:
            X_train, y_train, full_distrib_train, X_val, y_val, full_distrib_val = load_pickled_ds('/home/al/projects/global_rs_project/data/pickled_data/raw_train_test_splitted_s1_60.pkl')
            
        else:
            X_train, y_train, full_distrib_train, X_val, y_val, full_distrib_val = load_pickled_ds('/home/al/projects/global_rs_project/data/pickled_data/raw_train_test_splitted_s2_60.pkl')

        
        val_dataset = NumpyDataset(X_val, y_val, sentinel_number=sentinel_number, band_order=band_order, requested_indices=requested_indices)
        train_dataset = NumpyDataset(X_train, y_train, sentinel_number=sentinel_number, band_order=band_order, requested_indices=requested_indices)

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        
        wandb.config.update({
            "batch_size": bs,
            "learning_rate": lr,
            "required_indixes": ';'.join(requested_indices),
            "Sentinel": sentinel_number,
            "model_name": model_name,
            "decay_rate": decay_lr,
            "weigth_decay": weight_decay,
            "step2decay": step2decay,
            "experiment_name": experiment_name,
            "seed": seed
        }, allow_val_change=True)

        
        train_and_evaluate(
            experiment_name=experiment_name,
            model_name=model_name,
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device, 
            num_epochs=num_epochs[0], 
            lr=lr,
            decay_lr=decay_lr,
            step2decay=step2decay,
            seed=seed,
            weight_decay=weight_decay,
            loss_fn=CrossEntropyLoss,
            metric_fn=accuracy
        )

    run.finish()

if __name__ == "__main__":
    cf_name = sys.argv.pop()
    print(cf_name) 
    dmain = hydra.main(version_base="1.3", config_path="conf", config_name=cf_name)(main)
    dmain()