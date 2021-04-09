
"""
Generate a single data object containing statistics for each post, 
associate comments, and the topic distribution
"""

## Script Meta Parameters
NUM_JOBS = 8

#####################
### Imports
#####################

## Standard Library
import os
import sys
import json
from datetime import datetime
from collections import Counter
from multiprocessing import Pool

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

#####################
### Globals
#####################

## Data Directory
DATA_DIR = "./data/"

#####################
### Helpers
#####################

def default_comment_metadata():
    """

    """
    metadata = {
            'n_response': 0,
            'n_unique_responders': 0,
            'received_physician_response': False,
            'received_physician_in_training_response': False,
            'received_non_physician_provider_response': False,
            'received_non_physician_provider_in_training_response': False,
            'min_created_utc': np.nan,
            'min_created_utc_physician': np.nan,
            'min_created_utc_non_physician_provider': np.nan,
            'min_created_utc_physician_in_training': np.nan,
            'min_created_utc_non_physician_provider_in_training': np.nan,
            'received_any_provider_response': False,
            'min_created_utc_any_provider_response': np.nan}
    return metadata

def count_prior_user_submissions(submission_metadata):
    """

    """
    submission_metadata = submission_metadata.sort_values("created_utc")
    author_counts = Counter()
    n_prior_submissions = np.zeros(submission_metadata.shape[0], dtype=int)
    for a, author in enumerate(submission_metadata["author"].tolist()):
        n_prior_submissions[a] = author_counts[author]
        author_counts[author] += 1
    submission_metadata["n_prior_submissions"] = n_prior_submissions
    submission_metadata = submission_metadata.drop("author",axis=1)
    return submission_metadata

def time_to_events(sub_metadata,
                   com_metadata):
    """

    """
    submission_created = datetime.fromtimestamp(sub_metadata.get("created_utc"))
    timings = {}
    for measure in [c for c in com_metadata.keys() if "min_created_utc" in c]:
        timing_measure = "time_to_{}".format(measure[16:])
        if np.isnan(com_metadata[measure]):
            timings[timing_measure] = None
        else:
            diff = (datetime.fromtimestamp(com_metadata[measure]) - submission_created).total_seconds() / 60
            timings[timing_measure] = diff
    return timings

#####################
### Data Loading
#####################

print("Loading Data Resources")

## Metadata
submission_metadata = pd.read_csv(f"{DATA_DIR}/processed/submission_metadata.csv",index_col=0)
comment_metadata = pd.read_csv(f"{DATA_DIR}/processed/comment_metadata.csv",index_col=0)

## Extend Submission Metadata
submission_metadata = count_prior_user_submissions(submission_metadata)

## Format Metadata into Dictionaries
submission_metadata = submission_metadata.to_dict(orient="index")
comment_metadata = comment_metadata.to_dict(orient="index")

## Topic Assignments
submission_topic_assignments = {}
with open(f"{DATA_DIR}/processed/models/topic_model/topic_assignments.json","r") as the_file:
    for line in the_file:
        line_data = json.loads(line)
        submission_topic_assignments[line_data["id"]] = line_data["topics"]

## Topic Metadata
topic_labels = [i.split() for i in open(f"{DATA_DIR}/processed/models/topic_model/topic_labels.txt","r")]
topic_labels = [[int(i[0]), i[1].lower().replace("_"," ").title()] for i in topic_labels]
topic_labels = pd.DataFrame(topic_labels, columns=["topic","label"])
topic_groups = {}
for _, row in topic_labels.iterrows():
    if row["label"] not in topic_groups:
        topic_groups[row["label"]] = set()
    topic_groups[row["label"]].add(row["topic"])
topic_labels = topic_labels.set_index("topic")["label"].to_dict()

#####################
### Merging
#####################
    
def merge_metadata(submission_id):
    """

    """
    ## Get Metadata
    sub_metadata = submission_metadata.get(submission_id, None)
    com_metadata = comment_metadata.get(submission_id,default_comment_metadata())
    ## Check for Missing Submission
    if sub_metadata is None:
        return None
    ## Get Topics
    sub_topics = submission_topic_assignments.get(submission_id, [])
    sub_topics_grouped = {}
    for topic, prop in sub_topics:
        topic_group = topic_labels[topic]
        if topic_group not in sub_topics_grouped:
            sub_topics_grouped[topic_group] = 0
        sub_topics_grouped[topic_group] += prop
    if len(sub_topics) > 0:
        max_topic = topic_labels[sorted(sub_topics, key=lambda x: x[1], reverse=True)[0][0]]
    else:
        max_topic = None
    ## Add Topics to Submission Metadata
    sub_metadata["max_topic"] = max_topic
    for group in topic_groups.keys():
        sub_metadata[group] = sub_topics_grouped.get(group,0.)
    ## Append To Comment Metadata
    comment_timings = time_to_events(sub_metadata, com_metadata)
    for c in list(com_metadata.keys()):
        if "min_created_utc" in c:
            _ = com_metadata.pop(c, None)
    com_metadata.update(comment_timings)
    return submission_id, sub_metadata, com_metadata

def extract_statistics(submission_metadata,
                       comment_metadata):
    """

    """
    ## Get Submission IDs
    submission_ids = list(submission_metadata.keys())
    ## Extract Statistics
    mp = Pool(NUM_JOBS)
    results = list(tqdm(mp.imap_unordered(merge_metadata, submission_ids),
                        desc="Merging Metadata",
                        file=sys.stdout,
                        total=len(submission_ids)))
    _ = mp.close()
    ## Filter and Sort
    results = list(filter(lambda r: r is not None, results))
    results = sorted(results, key=lambda x: x[1]["created_utc"])
    ## Parse
    submission_ids = [r[0] for r in results]
    X = [r[1] for r in results]
    Y = [r[2] for r in results]
    ## Format
    X = pd.DataFrame(X, index=submission_ids)
    Y = pd.DataFrame(Y, index=submission_ids)
    return X, Y

## Extract Statistics (Form Independent and Target Variables)
print("Forming Independent and Dependent Variables")
X, Y = extract_statistics(submission_metadata, comment_metadata)

## Cache Variables
print("Caching Variables")
X.to_csv(f"{DATA_DIR}/processed/independent.csv")
Y.to_csv(f"{DATA_DIR}/processed/dependent.csv")

print("Script Complete!")