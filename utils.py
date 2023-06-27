import os, ast, yaml, json, random, datetime
import argparse

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='MP-DocVQA framework')

    # Required
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to yml file with model configuration.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to yml file with dataset configuration.')

    # Optional
    parser.add_argument('--eval-start', action='store_true', default=True, help='Whether to evaluate the model before training or not.')
    parser.add_argument('--no-eval-start', dest='eval_start', action='store_false')

    # Overwrite config parameters
    parser.add_argument('-p', '--page-retrieval', type=str, help='Page retrieval set-up.')
    parser.add_argument('-bs', '--batch-size', type=int, help='DataLoader batch size.')
    parser.add_argument('-msl', '--max-sequence-length', type=int, help='Max input sequence length of the model.')
    parser.add_argument('--seed', type=int, help='Seed to allow reproducibility.')
    parser.add_argument('--save-dir', type=str, help='Seed to allow reproducibility.')

    # Flower
    parser.add_argument('--flower', action='store_true', default=False, help='Use FL Flower.')
    parser.add_argument('--sample_clients', type=int, help='Number of sampled clients during FL.')
    parser.add_argument('--num_rounds', type=int, help='Number of FL rounds.')
    # parser.add_argument('--client_sampling_probability', type=float, help='.')  # (Number of selected clients / total number of clients)
    parser.add_argument('--iterations_per_fl_round', type=int, help='Number of iterations per provider during each FL round.')
    parser.add_argument('--providers_per_fl_round', type=int, help='Number of groups (providers) sampled in each FL Round.')

    parser.add_argument('--use_dp', action='store_true', default=False, help='Add Differential Privacy noise.')
    parser.add_argument('--sensitivity', type=float, help='Upper bound of the contribution per group (provider).')
    parser.add_argument('--noise_multiplier', type=float, help='Noise multiplier.')

    return parser.parse_args()


def parse_multitype2list_arg(argument):
    if argument is None:
        return argument

    if '-' in argument and '[' in argument and ']' in argument:
        first, last = argument.strip('[]').split('-')
        argument = list(range(int(first), int(last)))
        return argument

    argument = ast.literal_eval(argument)

    if isinstance(argument, int):
        argument = [argument]

    elif isinstance(argument, list):
        argument = argument

    return argument


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def save_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f)


"""
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
"""


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def check_config(config):
    model_name = config.model_name.lower()

    if 'page_retrieval' not in config:
        config.page_retrieval = 'none'

    page_retrieval = config.page_retrieval.lower()
    if model_name not in ['hi-layoutlmv3', 'hi-lt5', 'hi-vt5'] and page_retrieval == 'custom':
        raise ValueError("'Custom' retrieval is not allowed for {:}".format(model_name))

    elif model_name in ['hi-layoutlmv3, hilt5', 'hi-lt5', 'hivt5', 'hi-vt5'] and page_retrieval in ['concat', 'logits']:
        raise ValueError("Hierarchical model {:} can't run on {:} retrieval type. Only 'oracle' and 'custom' are allowed.".format(model_name, page_retrieval))

    if page_retrieval == 'custom' and model_name not in ['hi-layoutlmv3', 'hi-lt5', 'hi-vt5']:
        raise ValueError("'Custom' page retrieval only allowed for Heirarchical methods ('hi-layoutlmv3', 'hi-lt5', 'hi-vt5').")

    elif page_retrieval in ['concat', 'logits'] and getattr(config, 'max_pages', None) is not None:
        print("WARNING - Max pages ({:}) value is ignored for {:} page-retrieval setting.".format(getattr(config, 'max_pages'), page_retrieval))

    elif page_retrieval == 'none' and config.dataset_name not in ['SP-DocVQA', 'PFL-DocVQA']:
        print("Page retrieval can't be none for dataset '{:s}'. This is intended only for single page datasets. Please specify in the method config file the 'page_retrieval' setup to one of the following: [oracle, concat, logits, custom] ".format(config.dataset_name))

    if config.flower:
        assert config.fl_params.sample_clients <= config.fl_params.total_clients, "Number of sampled clients ({:d}) can't be greater than total number of clients ({:d})".format(config.fl_params.sample_clients, config.fl_params.total_clients)

    if 'save_dir' in config:
        if not config.save_dir.endswith('/'):
            config.save_dir = config.save_dir + '/'

        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
            os.makedirs(os.path.join(config.save_dir, 'results'))
            os.makedirs(os.path.join(config.save_dir, 'communication_logs'))

    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    config.experiment_name = "{:s}__{:}".format(config.model_name, experiment_date)

    return True


def load_config(args):
    model_config_path = "configs/models/{:}.yml".format(args.model)
    dataset_config_path = "configs/datasets/{:}.yml".format(args.dataset)
    model_config = parse_config(yaml.safe_load(open(model_config_path, "r")), args)
    dataset_config = parse_config(yaml.safe_load(open(dataset_config_path, "r")), args)
    training_config = model_config.pop('training_parameters')

    # Keep FL and DP parameters to move it to a lower level config (config.fl_config / config.dp_config)
    fl_config = model_config.pop('fl_parameters') if 'fl_parameters' in model_config and args.flower else None
    dp_config = model_config.pop('dp_parameters') if 'dp_parameters' in model_config and args.use_dp else None
    fl_dp_keys = []
    # Update (overwrite) the config yaml with input args.
    if fl_config is not None:
        fl_config.update({k: v for k, v in args._get_kwargs() if k in fl_config and v is not None})
        fl_dp_keys.extend(fl_config.keys())

    if dp_config is not None:
        dp_config.update({k: v for k, v in args._get_kwargs() if k in dp_config and v is not None})
        fl_dp_keys.extend(dp_config.keys())

    # Merge config values and input arguments.
    config = {**dataset_config, **model_config, **training_config}
    config = config | {k: v for k, v in args._get_kwargs() if v is not None}

    # Remove duplicate keys
    config.pop('model')
    config.pop('dataset')
    [config.pop(k) for k in list(config.keys()) if (k in fl_dp_keys)]

    config = argparse.Namespace(**config)

    if fl_config is not None:
        config.fl_params = argparse.Namespace(**fl_config)

    if dp_config is not None:
        config.dp_params = argparse.Namespace(**dp_config)

        # config['group_sampling_probability'] = config['client_sampling_probability'] * 50 / 340  # (Number of selected clients / total number of clients) * (Number of selected groups / MIN(number of groups among the clients))
        config.dp_params.group_sampling_probability = config.dp_params.client_sampling_probability * config.dp_params.providers_per_fl_round / 340  # 0.1960  # config['client_sampling_probability'] * 50 / 340  # (Number of selected clients / total number of clients) * (Number of selected groups / MIN(number of groups among the clients))

    # Set default seed
    if 'seed' not in config:
        print("Seed not specified. Setting default seed to '{:d}'".format(42))
        config.seed = 42

    check_config(config)

    return config


def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get('includes', []):
        config = load_config(included_config_path, args) | config

    return config


def correct_alignment(context, answer, start_idx, end_idx):
    if context[start_idx: end_idx] == answer:
        return [start_idx, end_idx]

    elif context[start_idx - 1: end_idx] == answer:
        return [start_idx - 1, end_idx]

    elif context[start_idx: end_idx + 1] == answer:
        return [start_idx, end_idx + 1]

    else:
        print(context[start_idx: end_idx], answer)
        return None


def time_stamp_to_hhmmss(timestamp, string=True):
    hh = int(timestamp/3600)
    mm = int((timestamp-hh*3600)/60)
    ss = int(timestamp - hh*3600 - mm*60)

    time = "{:02d}:{:02d}:{:02d}".format(hh, mm, ss) if string else [hh, mm, ss]

    return time


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
