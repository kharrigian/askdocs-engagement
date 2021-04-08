
"""
Identify Health Topics in Submissions
"""

####################
### Configuration
####################

## Script Meta Parameters
NUM_JOBS = 8
MODEL_DIR = "./data/processed/models/topic_model/"
RANDOM_SEED = 42
CACHE_TOP_K = 50

## Model Parameters
MODEL_N_ITER = 5000
INITIAL_K = 100
ALPHA_PRIOR = 0.1
ETA_PRIOR = 0.01
GAMMA_PRIOR = 0.01

## Vocabulary Parameters
MAX_N_GRAM = 3
PHRASE_THRESHOLD = 10
MIN_VOCAB_DF = 10
MIN_VOCAB_CF = 25
MAX_VOCAB_SIZE = 500000
RM_TOP_VOCAB = 250

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
import tomotopy as tp
from scipy import sparse
import matplotlib.pyplot as plt
from gensim.models import Phrases
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction import DictVectorizer

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
              exclude_ignorable_accounts=True,
              length_only=False):
    """

    """
    data = []
    length = 0
    with gzip.open(filename,"r") as the_file:
        for line_data in json.load(the_file):
            if exclude_ignorable_accounts and line_data.get("author") in IGNORABLES:
                continue
            if min_date is not None and line_data.get("created_utc") < min_date:
                continue
            if max_date is not None and line_data.get("created_utc") >= max_date:
                continue
            if length_only:
                length += 1
                continue
            if filters:
                line_data = dict((f, line_data.get(f,None)) for f in filters)
            data.append(line_data)
    if length_only:
        return length
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
        self._initialize_filenames()
    
    def _initialize_filenames(self):
        """

        """
        filenames = []
        wrapper = lambda x: x
        if self.verbose:
            print("Isolating Nonempty Files")
            wrapper = lambda x: tqdm(x, total=len(x), desc="Filesize Filter", file=sys.stdout)
        for filename in wrapper(self.filenames):
            lf = load_data(filename,
                           min_date=self.min_date,
                           max_date=self.max_date,
                           exclude_ignorable_accounts=self.exclude_ignorable_accounts,
                           length_only=True)
            if lf > 0:
                filenames.append(filename)
        self.filenames = filenames

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
                   verbose=False,
                   model_dir=None):
    """

    """
    ## Look for Existing Phrasers
    if model_dir is not None:
        try:
            return load_phrasers(model_dir)
        except FileNotFoundError:
            pass
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
                         threshold=PHRASE_THRESHOLD,
                         delimiter=" ")]
    current_n = 2
    while current_n < MAX_N_GRAM:
        print(f"Learning {current_n+1}-grams")
        phrasers.append(Phrases(phrasers[-1][vocab_stream],
                                max_vocab_size=MAX_VOCAB_SIZE,
                                threshold=PHRASE_THRESHOLD,
                                delimiter=" "))
        current_n += 1
        ngrams.append(current_n)
    print("Vocabulary Learning Complete")
    if model_dir is not None:
        _ = cache_phrasers(phrasers, ngrams, model_dir)
    return phrasers, ngrams

def cache_phrasers(phrasers,
                   ngrams,
                   model_dir):
    """

    """
    if not os.path.exists(f"{model_dir}/phrasers/"):
        _ = os.makedirs(f"{model_dir}/phrasers/")
    for phraser, ngram in zip(phrasers, ngrams):
        phraser_file = f"{model_dir}/phrasers/{ngram}.phraser"
        phraser.save(phraser_file)

def load_phrasers(model_dir):
    """

    """
    phraser_files = sorted(glob(f"{model_dir}/phrasers/*.phraser"))
    if len(phraser_files) == 0:
        raise FileNotFoundError("No phrasers found in the given model directory.")
    phrasers = []
    for pf in phraser_files:
        pf_ngram = int(os.path.basename(pf).split(".phraser")[0])
        pf_phraser = Phrases.load(pf)
        phrasers.append((pf_ngram, pf_phraser))
    phrasers = sorted(phrasers, key=lambda x: x[0])
    ngrams = [p[0] for p in phrasers]
    phrasers = [p[1] for p in phrasers]
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
                   verbose=True,
                   model_dir=None):
    """

    """
    ## Check for Existing Document Term
    if model_dir is not None:
        try:
            return load_document_term(model_dir)
        except FileNotFoundError:
            pass
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
    print("Final Vocabulary Size: {}".format(X.shape[1]))
    ## Cache Document Term
    if model_dir is not None:
        _ = cache_document_term(X, post_ids, vocab, model_dir)
    return X, post_ids, vocab

def cache_document_term(X,
                        post_ids,
                        vocabulary,
                        model_dir):
    """

    """
    ## Filenames
    X_filename = f"{model_dir}/data.npz"
    post_ids_filename = f"{model_dir}/posts.txt"
    vocabulary_filename = f"{model_dir}/vocabulary.txt"
    for obj, filename in zip([post_ids, vocabulary],[post_ids_filename,vocabulary_filename]):
        with open(filename,"w") as the_file:
            for item in obj:
                the_file.write(f"{item}\n")
    sparse.save_npz(X_filename, X)

def load_document_term(model_dir):
    """

    """
    ## Establish Filenames and Check Existence
    X_filename = f"{model_dir}/data.npz"
    post_ids_filename = f"{model_dir}/posts.txt"
    vocabulary_filename = f"{model_dir}/vocabulary.txt"
    for f in [X_filename, post_ids_filename, vocabulary_filename]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Could not find {f}")
    ## Load
    X = sparse.load_npz(X_filename)
    post_ids = [i.strip() for i in open(post_ids_filename,"r")]
    vocabulary = [i.strip() for i in open(vocabulary_filename,"r")]
    return X, post_ids, vocabulary

def generate_corpus(X, vocabulary=None):
    """

    """
    corpus = tp.utils.Corpus()
    for x in tqdm(X, total=X.shape[0], desc="Generating Corpus"):
        xn = x.nonzero()[1]
        xc = [[i]*j for i, j in zip(xn, x[0, xn].A[0])]
        xc = [i for j in xc for i in j]
        if vocabulary is not None:
            xv = [vocabulary[i] for i in xc]
        else:
            xv = list(map(str, xc))
        corpus.add_doc(xv)
    return corpus

def main():
    """

    """
    ## Establish Model Directory
    if not os.path.exists(MODEL_DIR):
        _ = os.makedirs(MODEL_DIR)
    ## Filenames
    submission_filenames = sorted(glob(f"{DATA_DIR}raw/AskDocs/submissions/*.json.gz"))
    ## Try To Load Phrasers, Or Learn Them as Fallback
    phrasers, ngrams = learn_phrasers(submission_filenames,
                                      verbose=False,
                                      model_dir=MODEL_DIR)
    ## Get Vectorized Representation
    X, post_ids, vocabulary = vectorize_data(filenames=submission_filenames,
                                             phrasers=phrasers,
                                             ngrams=ngrams,
                                             verbose=False,
                                             model_dir=MODEL_DIR)
    id2word = dict(zip(range(X.shape[1]), vocabulary))
    ## Drop Examples Without Vocabulary
    sample_mask = np.nonzero(X.getnnz(axis=1) > 0)[0]
    X = X[sample_mask]
    post_ids = [post_ids[sm] for sm in sample_mask]
    ## Transform Matrix to Vocabulary
    corpus = generate_corpus(X, vocabulary)
    ## Initialize Model
    model = tp.HDPModel(corpus=corpus,
                        initial_k=INITIAL_K,
                        alpha=ALPHA_PRIOR,
                        eta=ETA_PRIOR,
                        gamma=GAMMA_PRIOR,
                        seed=RANDOM_SEED)
    ## Fit Model using Gibbs Sampler
    print("Beginning Model Training")
    params = np.zeros((MODEL_N_ITER, 5))
    for iteration in tqdm(range(MODEL_N_ITER), total=MODEL_N_ITER, desc="MCMC", file=sys.stdout):
        model.train(1, workers=NUM_JOBS)
        params[iteration] = np.array([model.k, model.live_k, model.alpha, model.gamma, model.ll_per_word])
    live_topics = [i for i in range(model.k) if model.is_live_topic(i)]
    params = pd.DataFrame(params, columns=["k","live_k","alpha","gamma","ll"])
    params.to_csv(f"{MODEL_DIR}/hdp.mcmc.csv",index=False)
    ## Trace Plots
    fig, ax = plt.subplots(2, 2, figsize=(10,5.6),sharex=True)
    ax[0,0].plot(params["k"], label="K", color="C0", alpha=0.8)
    ax[0,0].plot(params["live_k"], label="Active K", color="C1", alpha=0.8)
    ax[0,1].plot(params["alpha"], color="C2", alpha=0.8)
    ax[1,0].plot(params["gamma"], color="C3", alpha=0.8)
    ax[1,1].plot(params["ll"], color="black", alpha=0.8)
    ax[0,0].legend(loc="lower right", fontsize=13)
    for i in range(2):
        ax[1,i].set_xlabel("MCMC Iteration")
        for j in range(2):
            ax[i,j].spines["right"].set_visible(False)
            ax[i,j].spines["top"].set_visible(False)
            ax[i,j].tick_params(labelsize=10)
    ax[0,0].set_title("# Components")
    ax[0,1].set_title("$\\alpha$")
    ax[1,0].set_title("$\\gamma$")
    ax[1,1].set_title("Log-Likelihood")
    fig.tight_layout()
    fig.savefig(f"{MODEL_DIR}trace.png",dpi=300)
    plt.close(fig)
    ## Save the model
    _ = model.save(f"{MODEL_DIR}/hdp.model")
    _ = model.summary(file=open(f"{MODEL_DIR}/hdp.summary.txt","w"), topic_word_top_n=20)
    with open(f"{MODEL_DIR}/hdp.summary.txt","a") as the_file:
        the_file.write("Live Topics: {}".format(", ".join(list(map(str, live_topics)))))
    ## Extract Topic Distribution
    print("Caching Topic Distribution")
    topic_word_dist = np.vstack([model.get_topic_word_dist(topic) for topic in live_topics])
    topic_terms = []
    for i, dist in enumerate(topic_word_dist):
        dist_sorting = np.argsort(dist)[::-1][:CACHE_TOP_K]
        dist_sorting_vocab = [[vocabulary[d], float(dist[d])] for d in dist_sorting]
        topic_terms.append({"topic":i, "terms":dist_sorting_vocab})
    with open(f"{MODEL_DIR}topic_terms.json","w") as the_file:
        for tt in topic_terms:
            _ = the_file.write(f"{json.dumps(tt)}\n")
    ## Topic Assignments
    print("Caching Topic Assignments")
    doc_topic_dist = np.vstack([doc.get_topic_dist() for doc in model.docs])[:,live_topics]
    topic_assignments = []
    for post_id, doc in zip(post_ids, doc_topic_dist):
        doc_topics = [[int(d), float(doc[d])] for d in doc.nonzero()[0]]
        topic_assignments.append({"id":post_id, "topics":doc_topics})
    with open(f"{MODEL_DIR}topic_assignments.json","w") as the_file:
        for ta in topic_assignments:
            _ = the_file.write(f"{json.dumps(ta)}\n")
    print("Script Complete.")

##################
### Execution
##################

if __name__ == "__main__":
    _ = main()


