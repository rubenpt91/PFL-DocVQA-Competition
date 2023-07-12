import os, shutil, random
import numpy as np
from utils import save_yaml

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
import models._model_utils as model_utils
import transformers.models.t5.modeling_t5
from transformers import T5Tokenizer, T5ForConditionalGeneration


class VT5(T5ForConditionalGeneration):
    def __init__(self, config):
        self.max_input_tokens = getattr(config, 'max_input_tokens', 512)

        t5_config = CustomT5Config.from_pretrained(config.model_weights)
        t5_config.visual_module_config = config.visual_module
        self.spatial_embedding = SpatialEmbeddings(t5_config).to(config.device)
        self.visual_embedding = VisualEmbeddings(t5_config).to(config.device)
        self.load_model(config.model_weights)

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None):
        bs = len(words)
        # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, words)]
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0  # To become [0, 0, 0, 0] array.

        # Get input_ids, attention_mask and boxes.
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # Tokenize the word and ignore eos_token
                input_ids.extend(tokenized_word)
                input_boxes.extend([box]*len(tokenized_word))  # Repeat the box for each token corresponding to the word.

            batch_input_ids.append(input_ids[:self.max_input_tokens-1] + [self.tokenizer.eos_token_id])  # Append the eos_token at the end.
            batch_input_boxes.append(np.concatenate([input_boxes[:self.max_input_tokens-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
            longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_input_tokens)

        # Convert to tensors and pad. Actually, a pad tensor is created and it's filled with corresponding values.
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx][:len(batch_input_boxes[batch_idx])])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1

        """
        context = [(' ').join(doc_words) for doc_words in words]
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        input_embeds = self.model.shared(tokens.input_ids)
        """

        # Send everything to GPU
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # Get semantic and spatial embeddings
        semantic_embedding = self.model.shared(tensor_input_ids)
        spatial_embedding = self.spatial_embedding(tensor_boxes)
        visual_embedding, visual_emb_mask = self.visual_embedding(images)

        # input_embeds = semantic_embedding
        input_embeds = torch.add(semantic_embedding, spatial_embedding)
        input_embeds = torch.cat([input_embeds, visual_embedding], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)

        """
        context = [' '.join(doc_words) for doc_words in words]
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        x = self.model.shared(tokens.input_ids)
        """

        # Tokenize answers
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None

        return input_embeds, tensor_attention_mask, labels

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']

        input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(question, words, boxes, images, answers)
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask) if return_pred_answer else None

        return outputs, pred_answers, pred_answers_conf

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)

        return pred_answers, pred_answers_conf

    def save_model(self, save_dir, round, kwargs, update_best):

        # Save language part.
        self.model.save_pretrained(os.path.join(save_dir, "round_{:d}.ckpt".format(round), "lm_model"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "round_{:d}.ckpt".format(round), "lm_model"))

        # Save spatial embedding.
        torch.save(self.spatial_embedding.state_dict(), os.path.join(save_dir, "round_{:d}.ckpt".format(round), "sp_emb".format(round)))

        # Save visual embedding.
        self.visual_embedding.image_model.save_pretrained(os.path.join(save_dir, "round_{:d}.ckpt".format(round), "vm_model"))
        self.visual_embedding.feature_extractor.save_pretrained(os.path.join(save_dir, "round_{:d}.ckpt".format(round), "vm_model"))
        torch.save(self.visual_embedding.visual_emb_matcher.state_dict(), os.path.join(save_dir, "round_{:d}.ckpt".format(round), "vm_model", "emb_mlp"))

        # Save config
        save_yaml(os.path.join(save_dir, "round_{:d}.ckpt".format(round), "experiment_config.yml"), kwargs)
        shutil.copy(os.path.join(save_dir, "round_{:d}.ckpt".format(round), "lm_model", "config.json"), os.path.join(save_dir, "round_{:d}.ckpt".format(round), "config.json"))

        if update_best:
            # Save language part.
            self.model.save_pretrained(os.path.join(save_dir, "best.ckpt", "lm_model"))
            self.tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt", "lm_model"))

            # Save spatial embedding.
            torch.save(self.spatial_embedding.state_dict(), os.path.join(save_dir, "best.ckpt", "sp_emb"))

            # Save visual embedding.
            self.visual_embedding.image_model.save_pretrained(os.path.join(save_dir, "best.ckpt", "vm_model"))
            self.visual_embedding.feature_extractor.save_pretrained(os.path.join(save_dir, "best.ckpt", "vm_model"))
            torch.save(self.visual_embedding.visual_emb_matcher.state_dict(), os.path.join(save_dir, "best.ckpt", "vm_model", "emb_mlp"))

            # Save config
            shutil.copy(os.path.join(save_dir, "best.ckpt", "lm_model", "config.json"), os.path.join(save_dir, "best.ckpt", "config.json"))
            save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), kwargs)

    def load_model(self, load_weights):

        # Load local weights
        if load_weights.endswith('.ckpt'):
            # Load language part.
            self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(load_weights, "lm_model"))
            self.model = T5ForConditionalGeneration.from_pretrained(os.path.join(load_weights, "lm_model"))

            # Load spatial embedding.
            self.spatial_embedding.load_state_dict(torch.load(os.path.join(load_weights, "sp_emb")))

            # Load visual embedding.
            self.visual_embedding.image_model.from_pretrained(os.path.join(load_weights, "vm_model"))
            self.visual_embedding.feature_extractor.from_pretrained(os.path.join(load_weights, "vm_model"))
            self.visual_embedding.visual_emb_matcher.load_state_dict(torch.load(os.path.join(load_weights, "vm_model", "emb_mlp")))

        # Load weights directly from Huggingface
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(load_weights)
            self.model = T5ForConditionalGeneration.from_pretrained(load_weights)
