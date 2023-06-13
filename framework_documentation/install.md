# Installation steps

1. git clone git@github.com:rubenpt91/PFL-DocVQA-Competition.git
2. Download dataset
3. conda env create -f environment.yml
4. conda activate mp_docvqa
5. Change paths in configs/datasets/DocILE-ELSA.yml to match dataset location
6. Mkdir for log and save paths and add paths to configs/models/T5.yml
7. python3 train.py -m T5 -d DocILE-ELSA_csc