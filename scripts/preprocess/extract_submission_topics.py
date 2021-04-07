
"""
Identify Health Topics in Submissions
"""

####################
### Configuration
####################

## Modeling Parameters
MAX_N_GRAM = 3
MIN_VOCAB_DF = 25
MIN_VOCAB_CF = 50
MAX_VOCAB_SIZE = 500000
RM_TOP_VOCAB = 250
TOPIC_SWEEP = [25,50,75,100,125,150,175,200]

####################
### Imports
####################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob
from datetime import datetime
from collections import Counter

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from gensim.models import LdaModel, LdaMulticore, Phrases, HdpModel
from gensim.models.phrases import Phraser
from gensim.models.callbacks import CallbackAny2Vec
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse

## Local Imports
try:
    path = os.path.abspath(os.path.dirname(__file__))
except:
    path = os.path.abspath("./scripts/preprocess/")
sys.path.append(path)
from tokenizer import Tokenizer, STOPWORDS

####################
### Globals
####################

## Data Directories
DATA_DIR = "./data/"

## Dates
MIN_DATE = "2017-01-01"; MIN_DATE = int(datetime.strptime(MIN_DATE, "%Y-%m-%d").timestamp())
MAX_DATE = "2021-01-01"; MAX_DATE = int(datetime.strptime(MAX_DATE, "%Y-%m-%d").timestamp())

## Accounts to Ignore
IGNORABLES = set([
    "AutoModerator",
    "[deleted]",
    "[removed]"
])

## Initialize Tokenizer
TOKENIZER = Tokenizer(stopwords=STOPWORDS,
                      keep_case=False,
                      negate_handling=True,
                      negate_token=False,
                      keep_punctuation=False,
                      keep_numbers=False,
                      expand_contractions=True,
                      keep_user_mentions=False,
                      keep_pronouns=True,
                      keep_url=False,
                      keep_hashtags=False,
                      keep_retweets=False,
                      emoji_handling=None,
                      strip_hashtag=False)

####################
### Functions
####################

def load_data(filename,
              filters=None,
              min_date=None,
              max_date=None,
              exclude_ignorable_accounts=True):
    """

    """
    data = []
    with gzip.open(filename,"r") as the_file:
        for line_data in json.load(the_file):
            if exclude_ignorable_accounts and line_data.get("author") in IGNORABLES:
                continue
            if min_date is not None and line_data.get("created_utc") < min_date:
                continue
            if max_date is not None and line_data.get("created_utc") >= max_date:
                continue
            if filters:
                line_data = dict((f, line_data.get(f,None)) for f in filters)
            data.append(line_data)
    return data
    
class PostStream(object):

    """

    """

    def __init__(self,
                 filenames,
                 min_date=None,
                 max_date=None,
                 stream_sentences=False,
                 text_fields=["selftext","title"],
                 exclude_ignorable_accounts=True,
                 return_metadata=False,
                 verbose=False):
        """

        """
        self.filenames = filenames
        self.min_date = min_date
        self.max_date = max_date
        self.stream_sentences = stream_sentences
        self.text_fields = set(text_fields)
        self.exclude_ignorable_accounts = exclude_ignorable_accounts
        self.return_metadata = return_metadata
        self.verbose = verbose

    def __iter__(self):
        """

        """
        wrapper = lambda x: x
        if self.verbose:
            wrapper = lambda x: tqdm(x, desc="PostStream", file=sys.stdout)
        for filename in wrapper(self.filenames):
            file_data = load_data(filename,
                                  filters=["created_utc","id"]+list(self.text_fields),
                                  min_date=self.min_date,
                                  max_date=self.max_date,
                                  exclude_ignorable_accounts=self.exclude_ignorable_accounts)
            for post in file_data:
                sentences = []
                if "title" in self.text_fields:
                    title_text = post.get("title")
                    if title_text is None:
                        title_text = ""
                    sentences.append(TOKENIZER.tokenize(post.get(title_text)))
                if "selftext" in self.text_fields:
                    selftext = post.get("selftext")
                    if selftext is None:
                        selftext = ""
                    for sentence in sent_tokenize(selftext):
                        sentences.append(TOKENIZER.tokenize(sentence))
                if not self.stream_sentences:
                    sentences = [[i for s in sentences for i in s]]
                sentences = list(filter(lambda i: len(i) > 0, sentences))
                for tokens in sentences:
                    if self.return_metadata:
                        yield post.get("id"), post.get("created_utc"), tokens
                    else:
                        yield tokens

def learn_phrasers(filenames,
                   verbose=False):
    """

    """
    ## Learn Vocabulary
    vocab_stream = PostStream(filenames,
                              min_date=MIN_DATE,
                              max_date=MAX_DATE,
                              stream_sentences=True,
                              text_fields=["selftext","title"],
                              verbose=verbose,
                              exclude_ignorable_accounts=True)
    print("Learning Initial Vocabulary (1-2 Grams)")
    ngrams = [2]
    phrasers =  [Phrases(sentences=vocab_stream,
                         max_vocab_size=MAX_VOCAB_SIZE,
                         threshold=100,
                         delimiter=" ")]
    current_n = 2
    while current_n < MAX_N_GRAM:
        print(f"Learning {current_n+1}-grams")
        phrasers.append(Phrases(phrasers[-1][vocab_stream],
                                max_vocab_size=MAX_VOCAB_SIZE,
                                threshold=100,
                                delimiter=" "))
        current_n += 1
        ngrams.append(current_n)
    print("Vocabulary Learning Complete")
    return phrasers, ngrams

def initialize_vectorizer(vocabulary):
    """
    Initialize a vectorizer that transforms a counter dictionary
    into a sparse vector of counts (with a uniform feature index)
    """
    ## Isolate Terms, Sort Alphanumerically
    ngram_to_idx = dict((t, i) for i, t in enumerate(vocabulary))
    ## Create Dict Vectorizer
    _count2vec = DictVectorizer(separator=":",dtype=int)
    _count2vec.vocabulary_ = ngram_to_idx.copy()
    rev_dict = dict((y, x) for x, y in ngram_to_idx.items())
    _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    return _count2vec

def vectorize_data(filenames,
                   phrasers,
                   ngrams,
                   verbose=True):
    """

    """
    ## Initialize Stream
    vector_stream = PostStream(filenames,
                               min_date=MIN_DATE,
                               max_date=MAX_DATE,
                               stream_sentences=True,
                               text_fields=["selftext","title"],
                               verbose=verbose,
                               return_metadata=True,
                               exclude_ignorable_accounts=True)
    ## Cache
    counts = {}
    for post_id, _, sentence in vector_stream:
        if post_id not in counts:
            counts[post_id] = Counter()
        counts[post_id] += Counter([i for i in phrasers[0][sentence] if i.count(" ") == ngrams[0]-2])
        for n, p in zip(ngrams, phrasers):
            counts[post_id] += Counter([i for i in p[sentence] if i.count(" ") == n - 1])
    ## Get Vocabulary
    vocab = set()
    for v in tqdm(counts.values(), total=len(counts)):
        vocab.update(v.keys())
    vocab = sorted(vocab)
    ## Vectorize
    count2vec = initialize_vectorizer(vocab)
    post_ids = list(counts.keys())
    X = sparse.vstack([count2vec.transform(counts[p]) for p in tqdm(post_ids,desc="Vectorization",file=sys.stdout)])
    ## Filter Vocabulary
    cf = X.sum(axis=0).A[0]
    df = (X!=0).sum(axis=0).A[0]
    top_k = set(np.argsort(cf)[-MAX_VOCAB_SIZE:-RM_TOP_VOCAB])
    vmask = np.logical_and(cf >= MIN_VOCAB_CF, df >= MIN_VOCAB_DF).nonzero()[0]
    vmask = list(filter(lambda v: v in top_k, vmask))
    vocab = [vocab[v] for v in vmask]
    X = X[:,vmask]
    return X, post_ids, vocab

def main():
    """

    """
    ## Filenames
    submission_filenames = sorted(glob(f"{DATA_DIR}raw/AskDocs/submissions/*.json.gz"))[-100:-95]
    ## Learn Phrasers
    phrasers, ngrams = learn_phrasers(submission_filenames, verbose=False)
    ## Get Vectorized Representation
    X, post_ids, vocabulary = vectorize_data(submission_filenames, phrasers, ngrams, True)
    id2word = dict(zip(range(X.shape[1]), vocabulary))
    ## Split Train/Test
    splits = np.array(list(map(lambda i: True if np.random.rand() < 0.8 else False, range(X.shape[0]))))
    X_train = X[splits.nonzero()[0]]
    X_test = X[np.logical_not(splits).nonzero()[0]]
    ## Fit Model
    results = []

    
    model = HdpModel(corpus=Sparse2Corpus(X_train, False),
                     id2word=id2word,
                     random_state=42,
                     max_chunks=(X_train.shape[0] // 256) * 100,
                     chunksize=256)
    model = model.suggested_lda_model()
    hdp_p_train = model.log_perplexity(Sparse2Corpus(X_train, False))
    hdp_p_test = model.log_perplexity(Sparse2Corpus(X_test, False))
    


    for j, k in enumerate(TOPIC_SWEEP):
        print(f"Training Model {j+1}/{len(TOPIC_SWEEP)} with {k} topics")
        ## Fit Model
        model = LdaMulticore(corpus=Sparse2Corpus(X_train, False),
                             id2word=id2word,
                             num_topics=k,
                             passes=1000,
                             workers=8,
                             alpha="symmetric",
                             eta="auto",
                             random_state=42,
                             iterations=100)
        ## Perplexity
        p_train = model.log_perplexity(Sparse2Corpus(X_train, False))
        p_test = model.log_perplexity(Sparse2Corpus(X_test, False))
        ## Topics
        model_topics = []
        for topic in range(k):
            topic_terms = [id2word[i[0]] for i in model.get_topic_terms(topic,20)]
            model_topics.append(topic_terms)
        ## Cache
        results.append((k, p_train, p_test, model_topics))
        