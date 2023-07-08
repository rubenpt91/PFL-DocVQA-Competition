# PFL-DocVQA Competition

This repository is intended to provide a base framework and method for the [PFL-DocVQA Competition](http://158.109.8.94/?ch=2&com=introduction).

<div style="text-align: justify;">
The objective of the Privacy Preserving Federated Learning Document VQA (PFL-DocVQA) competition is to develop privacy-preserving solutions for fine-tuning multi-modal language models for document understanding on distributed data.
We seek efficient federated learning solutions for finetuning a pre-trained generic Document Visual Question Answering (DocVQA) model on a new domain, that of invoice processing.

Automatically managing the information of document workflows is a core aspect of business intelligence and process automation.
Reasoning over the information extracted from documents fuels subsequent decision-making processes that can directly affect humans, especially in sectors such as finance, legal or insurance.
At the same time, documents tend to contain private information, restricting access to them during training.
This common scenario requires training large-scale models over private and widely distributed data.
</div>

## How to use
To set up and use the framework please check [How to use](framework_documentation/how_to_use.md#how-to-use) instructions.


## Dataset

[TODO] Dataset Description.

If you want to download the dataset, you can do so in the [ELSA Benchmarks Competition platform](http://158.109.8.94/?ch=2&com=introduction). For this framework, you will need to download the IMDBs (which contains processed QAs and OCR) and the images. All the downloads must be performed through the RRC portal.

| Dataset    | Link                                        |
|------------|---------------------------------------------|
| PFL-DocVQA | [Link](https://benchmarks.elsa-ai.eu/?ch=2) |


## PFL-DocVQA models weights

| Model    |                                  Weights link                                  | Parameters  |
|:---------|:------------------------------------------------------------------------------:|:-----------:|
| VT5 base | [weights](https://datasets.cvc.uab.es/elsa/PFL-DocVQA/vt5_mp-docvqa.ckpt.zip)  |             | 

## Metrics

**Average Normalized Levenshtein Similarity (ANLS)** <br>
The standard metric for text-based VQA tasks (ST-VQA and DocVQA). It evaluates the method's reasoning capabilities while smoothly penalizes OCR recognition errors. <br>
Check [Scene Text Visual Question Answering](https://arxiv.org/abs/1905.13648) for more details.
