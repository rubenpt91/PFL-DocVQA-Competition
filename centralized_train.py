from torch.utils.data import DataLoader

from build_utils import build_dataset, build_optimizer
from checkpoint import save_model
from datasets.BaseDataset import collate_fn
from eval import evaluate
from logger import Logger
from metrics import Evaluator
from utils import seed_everything


def train(model, config):

    epochs = config.train_epochs
    # device = config.device
    seed_everything(config.seed)

    evaluator = Evaluator(case_sensitive=False)
    logger = Logger(config)
    logger.log_model_parameters(model)

    train_dataset = build_dataset(config, 'train')
    val_dataset   = build_dataset(config, 'val')

    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

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
        _ = fl_train(train_data_loader, model, optimizer, lr_scheduler, evaluator, logger)
        accuracy, anls, ret_prec, _, _ = evaluate(val_data_loader, model, evaluator, config)

        is_updated = evaluator.update_global_metrics(accuracy, anls, epoch_ix)
        logger.log_val_metrics(accuracy, anls, ret_prec, update_best=is_updated)
        save_model(model, epoch_ix, config, update_best=is_updated)