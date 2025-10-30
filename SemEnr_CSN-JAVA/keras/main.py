# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import random
import traceback
import pickle
import argparse
import threading
import tables
import codecs
import logging
import math
from math import log

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scipy.stats import rankdata

import configs
from configs import get_config
from utils import cos_np, normalize, cos_np_for_normalized
from models import *
from models import JointEmbeddingModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

# -----------------------------
# Environment
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silence TF C++ logs

# -----------------------------
# GPU configuration for TF 2.x
# - Prefer memory growth (safe & simple)
# - If you truly need a hard cap, consider CUDA_VISIBLE_DEVICES with MIG or TF logical
#   device configs (requires fixed MB values).
# -----------------------------
def configure_gpu_memory_growth():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Enabled GPU memory growth on %d GPU(s).", len(gpus))
        else:
            logger.info("No GPU detected by TensorFlow.")
    except Exception as e:
        logger.warning("Failed to configure GPU memory growth: %s", e)

configure_gpu_memory_growth()


class CodeSearcher:
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self.path = self.conf.get('workdir', '../data/github/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())

        self.vocab_tokens = self.load_pickle(self.path + self.data_params['vocab_tokens'])
        self.vocab_desc = self.load_pickle(self.path + self.data_params['vocab_desc'])

        self._eval_sets = None

        self._code_reprs = None
        self._code_base = None
        self._code_base_chunksize = 2000000

    def load_pickle(self, filename):
        return pickle.load(open(filename, 'rb'))

    ##### Data Set #####
    def load_training_data_chunk(self):
        logger.debug('Loading a chunk of training data..')
        logger.debug('tokens')
        chunk_tokens = pickle.load(open(self.path + self.data_params['train_tokens'], 'rb'))
        logger.debug('desc')
        chunk_descs = pickle.load(open(self.path + self.data_params['train_desc'], 'rb'))
        logger.debug('sim_desc')
        chunk_sim_descs = pickle.load(open(self.path + self.data_params['train_sim_desc'], 'rb'))
        return chunk_tokens, chunk_sim_descs, chunk_descs

    def load_valid_data_chunk(self):
        logger.debug('Loading a chunk of validation data..')
        logger.debug('tokens')
        chunk_tokens = pickle.load(open(self.path + self.data_params['valid_tokens'], 'rb'))
        logger.debug('desc')
        chunk_descs = pickle.load(open(self.path + self.data_params['valid_desc'], 'rb'))
        logger.debug('sim_desc')
        chunk_sim_descs = pickle.load(open(self.path + self.data_params['valid_sim_desc'], 'rb'))
        return chunk_tokens[:500], chunk_sim_descs[:500], chunk_descs[:500]

    def load_test_data_chunk(self):
        logger.debug('Loading a chunk of validation data..')
        logger.debug('tokens')
        chunk_tokens = pickle.load(open(self.path + self.data_params['valid_tokens'], 'rb'))
        logger.debug('desc')
        chunk_descs = pickle.load(open(self.path + self.data_params['valid_desc'], 'rb'))
        logger.debug('sim_desc')
        chunk_sim_descs = pickle.load(open(self.path + self.data_params['valid_sim_desc'], 'rb'))
        return chunk_tokens[:], chunk_sim_descs[:], chunk_descs[:]

    def load_use_data(self):
        logger.info('Loading use data..')
        logger.info('tokens')
        tokens = pickle.load(open(self.path + self.data_params['valid_tokens'], 'rb'))
        logger.debug('sim_desc')
        sim_desc = pickle.load(open(self.path + self.data_params['valid_sim_desc'], 'rb'))
        return tokens, sim_desc

    def load_codebase(self):
        """load codebase
        codefile: h5 file that stores raw code
        """
        logger.info('Loading codebase ...')
        if self._code_base is None:
            codebase = []
            codes = codecs.open(self.path + self.data_params['use_codebase'], encoding='utf8', errors='replace').readlines()
            # use codecs to read in case of encoding problem
            for i in range(0, len(codes)):
                codebase.append(codes[i])
            self._code_base = codebase

    ### Results Data ###
    def load_code_reprs(self):
        logger.debug('Loading code vectors (chunk size={})..'.format(self._code_base_chunksize))
        if self._code_reprs is None:
            """reads vectors (2D numpy array) from a hdf5 file"""
            codereprs = []
            h5f = tables.open_file(self.path + self.data_params['use_codevecs'])
            vecs = h5f.root.vecs
            for i in range(0, len(vecs), self._code_base_chunksize):
                codereprs.append(vecs[i:i + self._code_base_chunksize])
            h5f.close()
            self._code_reprs = codereprs
        return self._code_reprs

    def save_code_reprs(self, vecs):
        npvecs = np.array(vecs)
        fvec = tables.open_file(self.path + self.data_params['use_codevecs'], 'w')
        atom = tables.Atom.from_dtype(npvecs.dtype)
        filters = tables.Filters(complib='blosc', complevel=5)
        ds = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape, filters=filters)
        ds[:] = npvecs
        fvec.close()

    ##### Converting / reverting #####
    def convert(self, vocab, words):
        """convert words into indices"""
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [vocab.get(w, 0) for w in words]

    def revert(self, vocab, indices):
        """revert indices into words"""
        ivocab = dict((v, k) for k, v in vocab.items())
        return [ivocab.get(i, 'UNK') for i in indices]

    ##### Padding #####
    def pad(self, data, len=None):
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Model Loading / saving #####
    def save_model_epoch(self, model, epoch):
        target_dir = os.path.join(self.path, 'models', self.model_params['model_name'])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        model.save(f"{target_dir}/epo{epoch:d}_sim.h5", overwrite=True)

    def load_model_epoch(self, model, epoch):
        model.load(f"{self.path}models/{self.model_params['model_name']}/epo{epoch:d}_sim.h5")

    ##### Training #####
    def train(self, model):
        if self.train_params.get('reload', 0) > 0:
            self.load_model_epoch(model, self.train_params['reload'])
        valid_every = self.train_params.get('valid_every', None)
        save_every = self.train_params.get('save_every', None)
        batch_size = self.train_params.get('batch_size', 128)
        nb_epoch = self.train_params.get('nb_epoch', 50)
        split = self.train_params.get('validation_split', 0)

        val_loss = {'loss': 1., 'epoch': 0}
        BestMRR = 0
        BestEpoch = 0
        f1 = open('results/training_results.txt', 'a', encoding='utf-8', errors='ignore')
        for i in range(self.train_params.get('reload', 0) + 1, nb_epoch):
            print('Epoch %d :: \n' % i, end='')
            logger.debug('loading data chunk..')
            chunk_tokens, chunk_sim_desc, chunk_descs = self.load_training_data_chunk()
            logger.debug('padding data..')
            chunk_padded_tokens = self.pad(chunk_tokens, self.data_params['tokens_len'])
            chunk_padded_sim_desc = self.pad(chunk_sim_desc, self.data_params['sim_desc_len'])
            chunk_padded_good_descs = self.pad(chunk_descs, self.data_params['desc_len'])
            chunk_bad_descs = [desc for desc in chunk_descs]
            random.shuffle(chunk_bad_descs)
            chunk_padded_bad_descs = self.pad(chunk_bad_descs, self.data_params['desc_len'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min')
            hist = model.fit([chunk_padded_tokens, chunk_padded_sim_desc, chunk_padded_good_descs, chunk_padded_bad_descs],
                             epochs=1, batch_size=batch_size, validation_split=split, callbacks=[early_stopping])
            if hist.history.get('val_loss', [float('inf')])[0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if valid_every is not None and i % valid_every == 0:
                succrate, mrr = self.valid(model, 10)
                if mrr > BestMRR:
                    BestMRR = mrr
                    BestEpoch = i
                print('BestEpoch={},BestMRR={}'.format(BestEpoch, BestMRR))
                f1.write('epoch={},sr1={}, MRR={}'.format(i, succrate, mrr) + '\n')
                f1.flush()

            if save_every is not None and i % save_every == 0:
                self.save_model_epoch(model, i)

    ##### Evaluation in the develop set #####
    def valid(self, model, topK):
        """
        validate in a code pool. 
        param:
            topK - top-K to consider
        """
        def SUCCRATE1(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index <= 1:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def SUCCRATE5(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index <= 5:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def SUCCRATE10(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index <= 10:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def ACC(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def MAP(real, predict):
            sum_ = 0.0
            for id_, val in enumerate(real):
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum_ = sum_ + (id_ + 1) / float(index + 1)
            return sum_ / float(len(real))

        def MRR(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum_ = sum_ + 1.0 / float(index + 1)
            return sum_ / float(len(real))

        def NDCG(real, predict):
            dcg = 0.0
            idcg = IDCG(len(real))
            for i, predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance = 1
                    rank = i + 1
                    dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
            return dcg / float(idcg)

        def IDCG(n):
            idcg = 0
            itemRelevance = 1
            for i in range(n):
                idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
            return idcg

        # load valid dataset
        if self._eval_sets is None:
            tokens, sim_desc, descs = self.load_valid_data_chunk()
            self._eval_sets = dict()
            self._eval_sets['tokens'] = tokens
            self._eval_sets['sim_desc'] = sim_desc
            self._eval_sets['descs'] = descs
        succrate1, succrate5, succrate10, acc, mrr, map_, ndcg = 0, 0, 0, 0, 0, 0, 0
        data_len = len(self._eval_sets['descs'])
        from tqdm import tqdm
        for i in tqdm(range(data_len)):
            desc = self._eval_sets['descs'][i]  # good desc
            descs = self.pad([desc] * data_len, self.data_params['desc_len'])
            tokens = self.pad(self._eval_sets['tokens'], self.data_params['tokens_len'])
            sim_desc = self.pad(self._eval_sets['sim_desc'], self.data_params['sim_desc_len'])
            n_results = topK
            sims = model.predict([tokens, sim_desc, descs], batch_size=1000).flatten()
            negsims = np.negative(sims)
            predict_origin = np.argsort(negsims)  # full ranking
            predict = predict_origin[:n_results]
            predict = [int(k) for k in predict]
            predict_origin = [int(k) for k in predict_origin]
            real = [i]
            succrate1 += SUCCRATE1(real, predict_origin)
            succrate5 += SUCCRATE5(real, predict_origin)
            succrate10 += SUCCRATE10(real, predict_origin)
            acc += ACC(real, predict)
            mrr += MRR(real, predict)
            map_ += MAP(real, predict)
            ndcg += NDCG(real, predict)
        succrate1 = succrate1 / float(data_len)
        succrate5 = succrate5 / float(data_len)
        succrate10 = succrate10 / float(data_len)
        acc = acc / float(data_len)
        mrr = mrr / float(data_len)
        map_ = map_ / float(data_len)
        ndcg = ndcg / float(data_len)
        print('SuccRate1={},SuccRate5={},SuccRate10={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(succrate1, succrate5, succrate10, acc, mrr, map_, ndcg))
        return succrate1, mrr

    def eval(self, model, topK):
        """
        validate in a code pool. 
        param:
            topK - top-K to consider
        """
        def SUCCRATE1(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index <= 1:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def SUCCRATE5(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index <= 5:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def SUCCRATE10(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index <= 10:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def ACC(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum_ = sum_ + 1
            return sum_ / float(len(real))

        def MAP(real, predict):
            sum_ = 0.0
            for id_, val in enumerate(real):
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum_ = sum_ + (id_ + 1) / float(index + 1)
            return sum_ / float(len(real))

        def MRR(real, predict):
            sum_ = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum_ = sum_ + 1.0 / float(index + 1)
            return sum_ / float(len(real))

        def NDCG(real, predict):
            dcg = 0.0
            idcg = IDCG(len(real))
            for i, predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance = 1
                    rank = i + 1
                    dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
            return dcg / float(idcg)

        def IDCG(n):
            idcg = 0
            itemRelevance = 1
            for i in range(n):
                idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
            return idcg

        # load test dataset
        if self._eval_sets is None:
            tokens, sim_desc, descs = self.load_test_data_chunk()
            self._eval_sets = dict()
            self._eval_sets['tokens'] = tokens
            self._eval_sets['sim_desc'] = sim_desc
            self._eval_sets['descs'] = descs
        succrate1, succrate5, succrate10, acc, mrr, map_, ndcg = 0, 0, 0, 0, 0, 0, 0
        data_len = len(self._eval_sets['descs'])
        f3 = open('results/eval_results_iter.txt', 'a', encoding='utf-8', errors='ignore')
        from tqdm import tqdm
        for i in tqdm(range(data_len)):
            desc = self._eval_sets['descs'][i]  # good desc
            descs = self.pad([desc] * data_len, self.data_params['desc_len'])
            tokens = self.pad(self._eval_sets['tokens'], self.data_params['tokens_len'])
            sim_desc = self.pad(self._eval_sets['sim_desc'], self.data_params['sim_desc_len'])
            n_results = topK
            sims = model.predict([tokens, sim_desc, descs], batch_size=1000).flatten()
            negsims = np.negative(sims)
            predict_origin = np.argsort(negsims)  # full ranking
            predict = predict_origin[:n_results]
            predict = [int(k) for k in predict]
            predict_origin = [int(k) for k in predict_origin]
            real = [i]
            succrate1 += SUCCRATE1(real, predict_origin)
            succrate5 += SUCCRATE5(real, predict_origin)
            succrate10 += SUCCRATE10(real, predict_origin)
            acc += ACC(real, predict)
            mrr += MRR(real, predict)
            map_ += MAP(real, predict)
            ndcg += NDCG(real, predict)
            if (i + 1) % 50 == 0:
                f3.write('SuccRate1={},SuccRate5={},SuccRate10={},MRR={}'.format(
                    succrate1 / float(i + 1), succrate5 / float(i + 1), succrate10 / float(i + 1), mrr / float(i + 1)) + '\n')
                f3.flush()
                print('SuccRate1={},SuccRate5={},SuccRate10={},MRR={}'.format(
                    succrate1 / float(i + 1), succrate5 / float(i + 1), succrate10 / float(i + 1), mrr / float(i + 1)))
        succrate1 = succrate1 / float(data_len)
        succrate5 = succrate5 / float(data_len)
        succrate10 = succrate10 / float(data_len)
        acc = acc / float(data_len)
        mrr = mrr / float(data_len)
        map_ = map_ / float(data_len)
        ndcg = ndcg / float(data_len)
        print('SuccRate1={},SuccRate5={},SuccRate10={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            succrate1, succrate5, succrate10, acc, mrr, map_, ndcg))
        f2 = open('results/eval_results.txt', 'a', encoding='utf-8', errors='ignore')
        f2.write('SuccRate1={},SuccRate5={},SuccRate10={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            succrate1, succrate5, succrate10, acc, mrr, map_, ndcg) + '\n')
        return succrate1, mrr

    def search(self, model, query, n_results=10):
        tokens, sim_desc = self.load_use_data()
        print('loading complete')
        padded_tokens = self.pad(tokens, self.data_params['tokens_len'])
        padded_sim_desc = self.pad(sim_desc, self.data_params['sim_desc_len'])

        data_len = len(tokens)
        desc = self.convert(self.vocab_desc, query)  # convert desc sentence to word indices
        padded_desc = self.pad([desc] * data_len, self.data_params['desc_len'])
        sims = model.predict([padded_tokens, padded_sim_desc, padded_desc], batch_size=1000).flatten()

        codes_out = []
        sims_out = []
        negsims = np.negative(sims)
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]
        codes_out = [self._code_base[k] for k in maxinds]
        sims_out = sims[maxinds]

        return codes_out, sims_out

    def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):
        # 1. compute similarity
        chunk_sims = cos_np_for_normalized(normalize(desc_repr), code_reprs)

        # 2. choose top results
        negsims = np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]
        chunk_codes = [self._code_base[i][k] for k in maxinds]
        chunk_sims = chunk_sims[0][maxinds]
        codes.extend(chunk_codes)
        sims.extend(chunk_sims)

    def postproc(self, codes_sims):
        codes_, sims_ = zip(*codes_sims)
        codes = [code for code in codes_]
        sims = [sim for sim in sims_]
        final_codes = []
        final_sims = []
        n = len(codes_sims)
        for i in range(n):
            is_dup = False
            for j in range(i):
                if codes[i][:80] == codes[j][:80] and abs(sims[i] - sims[j]) < 0.01:
                    is_dup = True
            if not is_dup:
                final_codes.append(codes[i])
                final_sims.append(sims[i])
        return zip(final_codes, final_sims)


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("--proto", choices=["get_config"], default="get_config",
                        help="Prototype config to use for config")
    parser.add_argument("--mode", choices=["train", "eval", "repr_code", "search"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluat models in a test set "
                             " The `repr_code/repr_desc` mode computes vectors"
                             " for a code snippet or a natural language description with a trained model.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = getattr(configs, args.proto)()
    codesearcher = CodeSearcher(conf)

    ##### Define model ######
    logger.info('Build Model')
    model = eval(conf['model_params']['model_name'])(conf)  # initialize the model
    model.build()
    optimizer = Adam(clipnorm=0.1)
    model.compile(optimizer=optimizer)

    if args.mode == 'train':
        codesearcher.train(model)

    elif args.mode == 'eval':
        # evaluate for a particular epoch
        # load model
        if conf['training_params']['reload'] > 0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.eval(model, 10)

    elif args.mode == 'repr_code':
        # load model
        if conf['training_params']['reload'] > 0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        vecs = codesearcher.repr_code(model)

    elif args.mode == 'search':
        # search code based on a desc
        if conf['training_params']['reload'] > 0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.load_codebase()
        resultsFile = codecs.open('/data/userName/SemEnr_CSN-JAVA/keras/results/search_results.txt', 'w', encoding='utf-8', errors='ignore')
        queriesFile = codecs.open('/data/userName/SemEnr_CSN-JAVA/keras/results/query.txt', "r", encoding='utf-8', errors='ignore')
        n_results = 10
        while 1:
            queries = queriesFile.readline().splitlines()
            if not queries:
                break
            query = queries[0]
            print(query)
            n_results = 10
            codes, sims = codesearcher.search(model, query, n_results)
            zipped = list(zip(codes, sims))
            zipped = sorted(zipped, reverse=True, key=lambda x: x[1])
            zipped = codesearcher.postproc(zipped)
            zipped = list(zipped)[:n_results]
            results = '\\n\\n'.join(map(str, zipped))  # combine the result into a returning string
            print(results)
            resultsFile.write('Query:{}'.format(query) + '\\n')
            resultsFile.write('Returned codes:{}'.format(results) + '\\n')

    # In TF 2.x there's no global Keras session to clear; use this to free resources if needed.
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
