Home Credit Default Risk
==============================

Kaggle competition: [https://www.kaggle.com/c/home-credit-default-risk/overview](https://www.kaggle.com/c/home-credit-default-risk/overview)


Overview:
------


`search_index` - a pipeline to build and test Similarity Index for the embedding model
`build_dataset [TBD]` - a pipeline to prepare custom dataset to train and test Embedding Model and Similarity Index
`embedding_model [TBD]` - a pipeline to prepare custom dataset to train and test Embedding Model
`data` - for data 
`models` - for models
`reports` - for  metrics and plots files 
`notebooks` - dir to keep Jupyter Notebooks (not committed)
`src` - for source code shared over projects


Setup:
------

## 1. Build environment:

For MacOS, install packages building required for Faiss:
```bash 
brew install libomp
brew install swig
```

Сreate and activate a virtual environment
```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install `image_embeddings` with fixed dependencies
```
pip install git+https://github.com/mnrozhkov/image_embeddings@dev
```


## Run 

Run `search_index` pipeline:
```bash
cd search_index && dvc exp run
```