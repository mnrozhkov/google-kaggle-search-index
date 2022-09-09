Google Universal Image Embedding
==============================

Kaggle competition: [https://www.kaggle.com/competitions/google-universal-image-embedding](https://www.kaggle.com/competitions/google-universal-image-embedding)


Overview:
------
`data` - for data 
`search_index` - a pipeline to build and test Similarity Index for the embedding model
    `models` - for models
    `src` - source code for `search_index` pipeline
    `reports` - for  metrics and plots files 
    
    `models` - for models
    
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


## Run 
Navigate to `search_index` and run commands to download data and embedding model before the main pipeline

### Load data 

Set env var `STORAGE_PATH` to a path to dataset storage directory, i.e.
```bash
export STORAGE_PATH=~/storage/kaggle-guie
```

Load a dataset
```bash
cd search_index 
dvc get https://github.com/iterative/google-kaggle-competition-data-pipeline datasets/kaggle_130k --rev pipeline_v2 -o $STORAGE_PATH
echo "baseline_split.zip" >> .gitignore
```

Load model

```bash 
dvc get https://github.com/iterative/google-kaggle-competition models/saved_model.pt -o models/
```

Run `search_index` pipeline:
```bash
dvc exp run
```

Notes:
- you may control how many images are used to build the search index by `data_load.num_examples` parameter in `params.yaml`. For example, `dvc exp run -S data_load.num_examples=300` build search index from 300 randomly selected images 