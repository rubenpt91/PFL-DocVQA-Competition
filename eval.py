import os, time, datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.BaseDataset import collate_fn

from logger import Logger
from metrics import Evaluator
from utils import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset

import flwr as fl
from utils_parallel import get_parameters, set_parameters, weighted_average


def evaluate(data_loader, model, evaluator, config):

    return_scores_by_sample = getattr(config, 'return_scores_by_sample', False)
    return_answers = getattr(config, 'return_answers', False)

    if return_scores_by_sample:
        scores_by_samples = {}
        total_accuracies = []
        total_anls = []
        total_ret_prec = []

    else:
        total_accuracies = 0
        total_anls = 0
        total_ret_prec = 0

    all_pred_answers = []
    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch['question_id'])
        with torch.no_grad():
            outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)
            # print(pred_answers)

        metric = evaluator.get_metrics(batch['answers'], pred_answers, batch.get('answer_type', None))

        if 'answer_page_idx' in batch and pred_answer_page is not None:
            ret_metric = evaluator.get_retrieval_metric(batch['answer_page_idx'], pred_answer_page)
        else:
            ret_metric = [0 for _ in range(bs)]

        if return_scores_by_sample:
            for batch_idx in range(bs):
                scores_by_samples[batch['question_id'][batch_idx]] = {
                    'accuracy': metric['accuracy'][batch_idx],
                    'anls': metric['anls'][batch_idx],
                    'ret_prec': ret_metric[batch_idx],
                    'pred_answer': pred_answers[batch_idx],
                    'pred_answer_conf': answer_conf[batch_idx],
                    'pred_answer_page': pred_answer_page[batch_idx] if pred_answer_page is not None else None
                }

        if return_scores_by_sample:
            total_accuracies.extend(metric['accuracy'])
            total_anls.extend(metric['anls'])
            total_ret_prec.extend(ret_metric)

        else:
            total_accuracies += sum(metric['accuracy'])
            total_anls += sum(metric['anls'])
            total_ret_prec += sum(ret_metric)

        if return_answers:
            all_pred_answers.extend(pred_answers)

    if not return_scores_by_sample:
        total_accuracies = total_accuracies/len(data_loader.dataset)
        total_anls = total_anls/len(data_loader.dataset)
        total_ret_prec = total_ret_prec/len(data_loader.dataset)
        scores_by_samples = []
    
    return total_accuracies, total_anls, total_ret_prec, all_pred_answers, scores_by_samples


def main_eval(config, local_rank=None):
    start_time = time.time()

    if config.distributed:
        config.global_rank = config.node_id * config.num_gpus + local_rank

        # Create distributed process group.
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=config.world_size,
            rank=config.global_rank
        )

    if config.device == 'cuda' and local_rank != None:
        config.local_rank = local_rank
        config.device = torch.device("cuda:{:d}".format(local_rank))

    config.return_scores_by_sample = True
    config.return_answers = True

    dataset = build_dataset(config, 'test')

    if config.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=config.world_size, rank=config.global_rank
        )
        pin_memory = True

    else:
        sampler = None
        pin_memory = False

    val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, sampler=sampler)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    accuracy_list, anls_list, answer_page_pred_acc_list, pred_answers, scores_by_samples = evaluate(val_data_loader, model, evaluator, config)

    if not config.distributed or config.global_rank == 0:
        accuracy, anls, answ_page_pred_acc = np.mean(accuracy_list), np.mean(anls_list), np.mean(answer_page_pred_acc_list)

        inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
        logger.log_val_metrics(accuracy, anls, answ_page_pred_acc, update_best=False)

        save_data = {
            "Model": config.model_name,
            "Model_weights": config.model_weights,
            "Dataset": config.dataset_name,
            "Page retrieval": getattr(config, 'page_retrieval', '-').capitalize(),
            "Inference time": inf_time,
            "Mean accuracy": accuracy,
            "Mean ANLS": anls,
            "Mean Retrieval precision": answ_page_pred_acc,
            "Scores by samples": scores_by_samples,
        }

        experiment_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(config.save_dir, 'results', "{:}_{:}_{:}__{:}.json".format(config.model_name, config.dataset_name, getattr(config, 'page_retrieval', '').lower(), experiment_date))
        save_json(results_file, save_data)

        print("Results correctly saved in: {:s}".format(results_file))


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        evaluator = Evaluator(case_sensitive=False)
        # loss, accuracy = test(self.model, self.valloader)
        total_accuracies, total_anls, total_ret_prec, all_pred_answers, scores_by_samples = evaluate(self.valloader, self.model, evaluator, config)  # data_loader, model, evaluator, **kwargs
        return float(0), len(self.valloader), {"accuracy": float(total_accuracies), "anls": total_anls}   # First parameter is loss.


def client_fn(node_id):
    """Create a Flower client representing a single organization."""
    model = build_model(config)
    dataset = build_dataset(config, 'test')
    val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return FlowerClient(model, val_data_loader, val_data_loader)


if __name__ == '__main__':

    # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9957'

    args = parse_args()
    config = load_config(args)

    if not config.flower:
        main_eval(config)

    else:
        NUM_CLIENTS = config.num_clients

        # Create FedAvg strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=0,  # Never sample less than 10 clients for training
            min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
            min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
            evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        )

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        client_resources = None
        # DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
        if config.device == "cuda":
            client_resources = {"num_gpus": 1}

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

    # Centralized evaluation
    # If fraction_evaluate is set to 0.0, federated evaluation will be disabled.
    # https://flower.dev/docs/evaluation.html


