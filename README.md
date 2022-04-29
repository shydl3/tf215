# SemEnr
Code Semantic Enrichment for Deep Code Search

## Dependency
> Tested in Ubuntu 16.04
* Python 2.7-3.6
* Keras 2.1.3 or newer
* Tensorflow-gpu 1.7.0


## Usage

   ### DataSets
  The processed datasets used in our paper will be found at https://pan.baidu.com/s/1mrVdCw-iz7ZY-wLIoI-bWg  password:75dl
  
  And the `/data` folder need be included by `/keras`. 
  
   
   ### Configuration
   
   Edit hyper-parameters and settings in `config.py`
   
   ### Train and Evaluate
   
   ```bash
   python main.py --mode train
   python main.py --mode eval
