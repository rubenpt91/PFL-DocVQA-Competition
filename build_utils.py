
import torch
import transformers

from transformers import get_scheduler


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=float(config.lr))
    num_training_steps = config.train_epochs * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config.warmup_iterations, num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def build_model(config):

    available_models = ['t5', 'vt5']
    if config.model_name.lower() == 't5':
        from models.T5 import T5
        model = T5(config)

    elif config.model_name.lower() == 'vt5':
        from models.VT5 import ProxyVT5 as VT5
        model = VT5(config)

    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose one of {:}".format(config.model_name, ', '.join(available_models)))

    if config.device == 'cuda' and config.data_parallel and torch.cuda.device_count() > 1:
        model.parallelize()

    model.model.to(config.device)
    return model


def build_dataset(config, split, node_id=None):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['use_images'] = True

    if node_id:
        dataset_kwargs['node_id'] = node_id

    # Build dataset
    if config.dataset_name == 'DocILE-ELSA':
        from datasets.DocILE_ELSA import DocILE_ELSA
        dataset = DocILE_ELSA(config.imdb_dir, config.images_dir, split, dataset_kwargs)

    else:
        raise ValueError

    return dataset


def build_provider_dataset(config, split, provider_to_doc, provider, node_id=None):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['use_images'] = True

    if node_id:
        dataset_kwargs['node_id'] = node_id


    # Build dataset

    indexes = provider_to_doc[provider]
    if config.dataset_name == 'DocILE-ELSA':
        from datasets.DocILE_ELSA import DocILE_ELSA
        dataset = DocILE_ELSA(config.imdb_dir, config.images_dir, split, dataset_kwargs, indexes)

    else:
        raise ValueError

    return dataset
