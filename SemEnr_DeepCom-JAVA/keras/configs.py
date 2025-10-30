
def get_config():   
    conf = {
        'workdir': './data/github/',
        'data_params':{
            #training data
            'train_tokens':'train.tokens.pkl',
            'train_desc':'train.desc.pkl',
            'train_sim_desc':'train_IR_code_desc.pkl',
            #valid data
            'valid_tokens':'test.tokens.pkl',
            'valid_desc':'test.desc.pkl',
            'valid_sim_desc':'test_IR_code_desc.pkl',
            #use data (computing code vectors)
            'use_codebase':'test_source.txt',#'use.rawcode.h5'  
            #results data(code vectors)            
            'use_codevecs':'use.codevecs.normalized.h5',#'use.codevecs.h5',         
                   
            #parameters
            'tokens_len':50,
            'desc_len': 30,
            'sim_desc_len': 30,
            'n_desc_words': 12317, # len(vocabulary) + 1
            'n_tokens_words': 37917,
            #vocabulary info
            'vocab_tokens':'vocab.tokens.pkl',
            'vocab_desc':'vocab.desc.pkl',
        },               
        'training_params': {
            'batch_size': 128,
            'chunk_size':100000,
            'nb_epoch': 400,
            'validation_split': 0.1,
            # 'optimizer': 'adam',
            #'optimizer': Adam(clip_norm=0.1),
            'valid_every': 1,
            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'save_every': 1,
            'reload':0, #that the model is reloaded from . If reload=0, then train from scratch
        },

        'model_params': {
            'model_name':'JointEmbeddingModel',
            'n_embed_dims': 100,
            'n_hidden': 400,#number of hidden dimension of code/desc representation
            # recurrent
            'n_lstm_dims': 200, # * 2
            'init_embed_weights_methname': None,#'word2vec_100_methname.h5', 
            'init_embed_weights_tokens': None,#'word2vec_100_tokens.h5',
            'init_embed_weights_sbt':None, 
            'init_embed_weights_desc': None,#'word2vec_100_desc.h5',
            'init_embed_weights_api':None,           
            'margin': 0.05,
            'sim_measure':'cos',#similarity measure: gesd, cosine, aesd
        }        
    }
    return conf




