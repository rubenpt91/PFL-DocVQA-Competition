# PFL-DocVQA Competition

This repository is inteded to provide a base framework and method for the [PFL-DocVQA Competition](http://158.109.8.94/?ch=2&com=introduction). 
The repository is a simplified branch of the [MP-DocVQA framework](https://github.com/rubenpt91/MP-DocVQA-Framework), even though we have reduced the complexity of this, traces of the original code may remain.


## How to use
To use the framework please check [How to use](framework_documentation/how_to_use.md#how-to-use) instructions.


## Dataset

[TODO] Dataset Description.

If you want to download the dataset, you can do so in the [ELSA Benchmarks Competition platform](http://158.109.8.94/?ch=2&com=introduction). For this framework, you will need to download the IMDBs (which contains processed QAs and OCR) and the images. All the downloads must be performed through the RRC portal.

| Dataset 		   | Link	                                                                          |
|--------------|--------------------------------------------------------------------------------|
| DocILE-ELSA | [Link](https://cvcuab-my.sharepoint.com/:u:/g/personal/rperez_cvc_uab_cat/EZEYu8DpG2FJhnh9LD0PPpUBmpaWP67QrGwYJJ4jo88cQQ?e=XPOmDC) |

In addition, if you need the images. We will need to download the original dataset from the [DocILE Challenge](https://rrc.cvc.uab.es/?ch=26downloads).

## DocILE-ELSA models weights

| Model 		   | Weights HF name								                                                                                                               | Parameters 	|
|:-----------|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------:|
| T5 base			 | [rubentito/vt5-base-docile-elsa]([rubentito/t5-base-docile-elsa](https://huggingface.co/rubentito/vt5-base-docile-elsa))		            |  			| 
| VT5 base			| [rubentito/vt5-base-docile-elsa]([rubentito/vt5-base-docile-elsa](https://huggingface.co/rubentito/vt5-base-docile-elsa))		           |  			| 

## Metrics

**Average Normalized Levenshtein Similarity (ANLS)** <br>
The standard metric for text-based VQA tasks (ST-VQA and DocVQA). It evaluates the method's reasoning capabilities while smoothly penalizes OCR recognition errors.
Check [Scene Text Visual Question Answering](https://arxiv.org/abs/1905.13648) for more details.
