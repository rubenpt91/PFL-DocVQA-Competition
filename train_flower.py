import os
import flwr as fl
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets.BaseDataset import collate_fn
from eval_flower import evaluate
from metrics import Evaluator
from build_utils import build_model, build_optimizer, build_dataset
from utils import parse_args, load_config, seed_everything
from utils_parallel import get_parameters, set_parameters, weighted_average

from logger import Logger
from checkpoint import save_model


def train_epoch(data_loader, model, optimizer, lr_scheduler, evaluator, logger):
    model.model.train()

    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        gt_answers = batch['answers']
        outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)
        loss = outputs.loss + outputs.ret_loss if hasattr(outputs, 'ret_loss') else outputs.loss

        total_loss += loss.item() / len(batch['question_id'])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

        metric = evaluator.get_metrics(gt_answers, pred_answers)

        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])

        log_dict = {
            'Train/Batch loss': outputs.loss.item(),
            'Train/Batch Accuracy': batch_acc,
            'Train/Batch ANLS': batch_anls,
            'lr': optimizer.param_groups[0]['lr']
        }

        if hasattr(outputs, 'ret_loss'):
            log_dict['Train/Batch retrieval loss'] = outputs.ret_loss.item()

        if 'answer_page_idx' in batch and None not in batch['answer_page_idx']:
            ret_metric = evaluator.get_retrieval_metric(batch.get('answer_page_idx', None), pred_answer_page)
            batch_ret_prec = np.mean(ret_metric)
            log_dict['Train/Batch Ret. Prec.'] = batch_ret_prec

        logger.logger.log(log_dict, step=logger.current_epoch * logger.len_dataset + batch_idx)

    return total_loss


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     np.seed(worker_seed)


def train(model, config):

    epochs = config.train_epochs
    # device = config.device
    batch_size = config.batch_size
    seed_everything(config.seed)

    evaluator = Evaluator(case_sensitive=False)
    logger = Logger(config)
    logger.log_model_parameters(model)

    train_dataset = build_dataset(config, 'train')
    val_dataset   = build_dataset(config, 'val')

    # g = torch.Generator()
    # g.manual_seed(config.seed)

    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    # train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=singledocvqa_collate_fn, worker_init_fn=seed_worker, generator=g)
    # val_data_loader   = DataLoader(val_dataset, batch_size=config.batch_size,  shuffle=False, collate_fn=singledocvqa_collate_fn, worker_init_fn=seed_worker, generator=g)

    logger.len_dataset = len(train_data_loader)
    optimizer, lr_scheduler = build_optimizer(model, length_train_loader=len(train_data_loader), config=config)

    config.return_scores_by_sample = False
    config.return_pred_answers = False

    if getattr(config, 'eval_start', False):
        logger.current_epoch = -1
        accuracy, anls, ret_prec, _, _ = evaluate(val_data_loader, model, evaluator, config)
        is_updated = evaluator.update_global_metrics(accuracy, anls, -1)
        logger.log_val_metrics(accuracy, anls, ret_prec, update_best=is_updated)

    for epoch_ix in range(epochs):
        logger.current_epoch = epoch_ix
        train_loss = train_epoch(train_data_loader, model, optimizer, lr_scheduler, evaluator, logger)
        accuracy, anls, ret_prec, _, _ = evaluate(val_data_loader, model, evaluator, config)

        is_updated = evaluator.update_global_metrics(accuracy, anls, epoch_ix)
        logger.log_val_metrics(accuracy, anls, ret_prec, update_best=is_updated)
        save_model(model, epoch_ix, config, update_best=is_updated)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, optimizer, lr_scheduler, evaluator, logger, config):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.logger = logger
        self.logger.log_model_parameters(self.model)
        self.config = config

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_loss = train_epoch(self.trainloader, self.model, self.optimizer, self.lr_scheduler, self.evaluator, self.logger)

        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        # loss, accuracy = test(self.model, self.valloader)
        # accuracy, anls, ret_prec, _, _ = evaluate(self.valloader, self.model, self.evaluator, return_scores_by_sample=False, return_pred_answers=False, **config)
        accuracy, anls, ret_prec, _, _ = evaluate(self.valloader, self.model, self.evaluator, config)  # data_loader, model, evaluator, **kwargs
        is_updated = self.evaluator.update_global_metrics(accuracy, anls, 0)
        self.logger.log_val_metrics(accuracy, anls, ret_prec, update_best=is_updated)

        return float(0), len(self.valloader), {"accuracy": float(accuracy), "anls": anls}


def client_fn(node_id):
    """Create a Flower client representing a single organization."""
    model = build_model(config)  # TODO Should already be in CUDA

    train_dataset = build_dataset(config, 'train', node_id)
    val_dataset = build_dataset(config, 'val')
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    optimizer, lr_scheduler = build_optimizer(model, length_train_loader=len(train_data_loader), config=config)
    evaluator = Evaluator(case_sensitive=False)
    logger = Logger(config=config)
    return FlowerClient(model, val_data_loader, val_data_loader, optimizer, lr_scheduler, evaluator, logger, config)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)

    if not config.flower:
        model = build_model(config)
        train(model, config)

    else:
        # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9957'

        NUM_CLIENTS = config.num_clients
        model = build_model(config)  # TODO Should already be in CUDA
        params = get_parameters(model)

        # Create FedAvg strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 50% of available clients for evaluation
            min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
            min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
            min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
            evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
            initial_parameters=fl.common.ndarrays_to_parameters(params),
        )

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        client_resources = None
        if config.device == "cuda":
            client_resources = {"num_gpus": 1}  # TODO Check number of GPUs

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

