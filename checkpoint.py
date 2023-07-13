import os
from utils import save_yaml


def save_model(model, epoch, kwargs, update_best=False):
    save_dir = os.path.join(kwargs.save_dir, 'checkpoints', "{:s}_{:s}".format(kwargs.model_name.lower(), kwargs.dataset_name.lower()))

    if getattr(kwargs, 'use_dp', False):
        save_dir += '_dp'

    # Models (VT5) that are not from Huggingface need to implement their own save_model function.
    if hasattr(model, 'save_model'):
        model.save_model(save_dir, epoch, kwargs, update_best)

    else:
        model.model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.processor if hasattr(model, 'processor') else None
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

        if hasattr(model.model, 'visual_embeddings'):
            model.model.visual_embeddings.feature_extractor.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

        save_yaml(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch), "experiment_config.yml"), kwargs)

        if update_best:
            model.model.save_pretrained(os.path.join(save_dir, "best.ckpt"))
            tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
            save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), kwargs)


"""
def load_model(base_model, ckpt_name, kwargs):
    load_dir = kwargs.save_dir
    base_model.model.from_pretrained(os.path.join(load_dir, ckpt_name))
"""
