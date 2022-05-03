# SemEnr
Code Semantic Enrichment for Deep Code Search

## Dependency
> Tested in Ubuntu 16.04
* Python 2.7-3.6
* Keras 2.1.3 or newer
* Tensorflow-gpu 1.7.0
* lucene 7.7.1


## Usage

   ### DataSets
  The datasets used in our paper will be found at: https://pan.baidu.com/s/19vAF889nbJgZ4NV3az8v1g password:m4o7
  
   ### Code Enrichment Module
   Build retrieval base: python Index.py
   
   Perform search: python Search.py
   
   ### Code Search Module
   
   #### Configuration
   
   Edit hyper-parameters and settings in `config.py`
   
   #### Train and Evaluate
   
   ```bash
   python main.py --mode train
   python main.py --mode eval
