
"""
Extract statistics for each submission based on the comments.
"""

## Script Meta Parameters
NUM_JOBS = 8

################
### Imports
################

## Standard Library
import os
import sys
import json 
import gzip
from glob import glob
from datetime import datetime
from functools import partial
from collections import Counter
from multiprocessing import Pool
from multiprocessing.dummy import Pool as Threads

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

####################
### Globals
####################

## Data
DATA_DIR = f"./data/"

## Dates
MIN_DATE = "2017-01-01"; MIN_DATE = int(datetime.strptime(MIN_DATE, "%Y-%m-%d").timestamp())
MAX_DATE = "2021-04-01"; MAX_DATE = int(datetime.strptime(MAX_DATE, "%Y-%m-%d").timestamp())

## Accounts to Ignore
IGNORABLES = set([
    "AutoModerator",
    "[deleted]",
    "[removed]"
])

## Flair Map
FLAIR_MAP = pd.read_csv(f"{DATA_DIR}/resources/flair_mapping.csv")
FLAIR_MAP["label"] = FLAIR_MAP.apply(lambda row:
        "none" if not row["is_health_professional"] else \
        "{}{}".format({True:"physician",False:"non_physician_provider"}[row["is_physician"]],
                      {True:"_in_training",False:""}[row["is_student"]])
        ,axis=1)
FLAIR_MAP = FLAIR_MAP.set_index("flair")["label"].to_dict()

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
            length += 1
            if length_only:
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
                 exclude_ignorable_accounts=True,
                 jobs=1,
                 kind="post"):
        """

        """
        self.filenames = filenames
        self.min_date = min_date
        self.max_date = max_date
        self.jobs = jobs
        self.exclude_ignorable_accounts = exclude_ignorable_accounts
        self.kind = kind
        assert self.kind in set(["post","file"])
        self._initialize_len()
    
    def _get_len(self,
                 filename):
        """

        """
        i, filename = filename
        return i, load_data(filename,
                            length_only=True,
                            min_date=self.min_date,
                            max_date=self.max_date,
                            exclude_ignorable_accounts=self.exclude_ignorable_accounts)

    def _initialize_len(self):
        """

        """
        print("Initializing Stream")
        if self.jobs == 1:
            lengths = list(tqdm(map(self._get_len, enumerate(self.filenames)),
                                total=len(self.filenames),
                                file=sys.stdout,
                                desc="Post Counter"))
        else:
            mp = Threads(self.jobs)
            lengths = list(tqdm(mp.imap_unordered(self._get_len, enumerate(self.filenames)),
                                total=len(self.filenames),
                                file=sys.stdout,
                                desc="Post Counter"))
            _ = mp.close()
        if self.kind == "post":
            self.len_ = sum([i[1] for i in lengths])
        else:
            self.len_ = len([i for i in lengths if i[1] > 0])
        self.filenames = [self.filenames[i[0]] for i in lengths if i[1] > 0]

    def __len__(self):
        """

        """
        return self.len_

    def __iter__(self):
        """

        """
        for f in self.filenames:
            data = load_data(f,
                             min_date=self.min_date,
                             max_date=self.max_date,
                             exclude_ignorable_accounts=self.exclude_ignorable_accounts,
                             length_only=False)
            if self.kind == "post":
                for post in data:
                    yield post
            elif self.kind == "file":
                yield data

def categorize_flair(post):
    """

    """
    return FLAIR_MAP.get(post.get("author_flair_text",None),"none")

def get_comment_metadata(posts):
    """

    """
    ## Add Provider Status
    submission_id = None
    for post in posts:
        post["provider_status"] = categorize_flair(post)
        if submission_id is None:
            submission_id = post["link_id"].split("_")[1]
        if post["link_id"].split("_")[1] != submission_id:
            raise ValueError("Found mismatched comments.")
    ## Overall Metadata
    comment_metadata = {
        "n_response":len(posts),
        "n_unique_responders":len(set(p.get("author") for p in posts)),
        "received_physician_response":any(p["provider_status"]=="physician" for p in posts),
        "received_physician_in_training_response":any(p["provider_status"]=="physician_in_training" for p in posts),
        "received_non_physician_provider_response":any(p["provider_status"]=="non_physician_provider" for p in posts),
        "received_non_physician_provider_in_training_response":any(p["provider_status"]=="non_physician_provider_in_training" for p in posts),
        "min_created_utc":min(p.get("created_utc") for p in posts),
        "min_created_utc_physician":min([p.get("created_utc") for p in posts if p["provider_status"]=="physician"]+[np.inf]),
        "min_created_utc_non_physician_provider":min([p.get("created_utc") for p in posts if p["provider_status"]=="non_physician_provider"]+[np.inf]),
        "min_created_utc_physician_in_training":min([p.get("created_utc") for p in posts if p["provider_status"]=="physician_in_training"]+[np.inf]),
        "min_created_utc_non_physician_provider_in_training":min([p.get("created_utc") for p in posts if p["provider_status"]=="non_physician_provider_in_training"]+[np.inf]),
    }
    comment_metadata["received_any_provider_response"] = any(comment_metadata[i] for i in ["received_physician_response","received_non_physician_provider_response","received_physician_in_training_response","received_non_physician_provider_in_training_response"])
    if comment_metadata["received_any_provider_response"]:
        comment_metadata["min_created_utc_any_provider_response"] = min(
            comment_metadata.get(i) for i in ["min_created_utc_physician","min_created_utc_non_physician_provider","min_created_utc_physician_in_training","min_created_utc_non_physician_provider_in_training"]
        )
    else:
        comment_metadata["min_created_utc_any_provider_response"] = np.inf
    for x, y in comment_metadata.items():
        if y == np.inf:
            comment_metadata[x] = None
    return submission_id, comment_metadata

def main():
    """

    """
    ## Get Comment Files and Initialize Stream
    comment_files = sorted(glob(f"{DATA_DIR}raw/AskDocs/comments/*.json.gz"))
    comment_stream = PostStream(comment_files,
                                min_date=MIN_DATE,
                                max_date=MAX_DATE,
                                exclude_ignorable_accounts=True,
                                jobs=NUM_JOBS * 2 - 1,
                                kind="file")
    ## Get Comment Metadata
    mp = Pool(NUM_JOBS)
    comment_metadata = dict(tqdm(mp.imap_unordered(get_comment_metadata, comment_stream),
                                 total=len(comment_stream),
                                 desc="Extracting Comment Metadata",
                                 file=sys.stdout))
    _ = mp.close()
    ## Format Metadata
    comment_metadata_df = pd.DataFrame.from_dict(comment_metadata, orient="index")
    comment_metadata_df.index.name = "id"
    ## Cache Metadata
    comment_metadata_df.to_csv(f"{DATA_DIR}/processed/comment_metadata.csv")
    print("Script Complete.")    

##################
### Execution
##################

# if __name__ == "__main__":
#     _ = main()