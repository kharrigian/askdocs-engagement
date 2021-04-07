
"""
Generate summary statistics for the dataset
"""

######################
### Imports
######################

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
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

######################
### Globals
######################

## Directories
DATA_DIR = f"./data/raw/AskDocs/"
PLOT_DIR = "./plots/summary_statistics/"
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

## Plotting Converters
_ = register_matplotlib_converters()

## Accounts to Ignore (Deleted, Automoderator)
IGNORABLES = set([
    "AutoModerator",
    "[deleted]",
    "[removed]"
])

## Dates
MIN_DATE = "2017-01-01"; MIN_DATE = int(datetime.strptime(MIN_DATE, "%Y-%m-%d").timestamp())
MAX_DATE = "2021-01-01"; MAX_DATE = int(datetime.strptime(MAX_DATE, "%Y-%m-%d").timestamp())

######################
### Helpers
######################

def load_data(filename,
              filters=None,
              exclude_ignorable_accounts=True):
    """

    """
    data = []
    with gzip.open(filename,"r") as the_file:
        for line_data in json.load(the_file):
            if exclude_ignorable_accounts and line_data.get("author") in IGNORABLES:
                continue
            if filters:
                line_data = dict((f, line_data.get(f,None)) for f in filters)
            data.append(line_data)
    return data

def count_activity(data_type,
                   min_date=None,
                   max_date=None):
    """

    """
    if data_type not in set(["submissions","comments"]):
        raise ValueError("`data_type` must be one of submissions or comments")
    ## Identify Filenames
    filenames = glob(f"{DATA_DIR}{data_type}/*.json.gz")
    filenames.sort()
    ## Initialize Caches
    counts = {}
    print(f"Counting {data_type}...")
    for filename in tqdm(filenames, desc="Progress", file=sys.stdout):
        data = load_data(filename, ["author","created_utc"])
        for post in data:
            if post.get("author",None) in IGNORABLES:
                continue
            if min_date is not None and post.get("created_utc",-1) < min_date:
                continue
            if max_date is not None and post.get("created_utc",np.inf) >= max_date:
                continue
            post_date = datetime.fromtimestamp(post.get("created_utc",None)).date()
            if post_date not in counts:
                counts[post_date] = Counter()
            counts[post_date][post.get("author")] += 1
    counts = pd.Series(counts).sort_index().to_frame("authors")
    counts["num_users"] = counts["authors"].map(len)
    counts["num_posts"] = counts["authors"].map(lambda x: sum(x.values()))
    return counts

######################
### Load Counts
######################

## Load Counts
submissions = count_activity("submissions", min_date=MIN_DATE, max_date=MAX_DATE)
comments = count_activity("comments", min_date=MIN_DATE, max_date=None)

## Reindex
index = pd.date_range(min(submissions.index.min(), comments.index.min()),
                      max(submissions.index.max(), comments.index.max()))
submissions = submissions.reindex(index)
comments = comments.reindex(index)

## Aggregated Counts By User
submissions_users = Counter(); comments_users = Counter()
for counts, support in zip([submissions_users, comments_users],[submissions,comments]):
    for row in tqdm(support["authors"].values, file=sys.stdout):
        if not isinstance(row, Counter):
            continue
        counts += row

######################
### Summary Stats and Visualizations
######################

## Summary Stats
fig, ax = plt.subplots(1,3,figsize=(10,5.8), sharey=True)
for i, (counts, name) in enumerate(zip([[comments_users],[submissions_users],[submissions_users, comments_users]],
                                  ["comments","submissions","combined"])):
    ## High Level
    all_counts = sum(counts, Counter())
    n_users = len(all_counts)
    n_posts = sum(all_counts.values())
    summary_str = f"{name.title()}:\n{n_users:,d} Users\n{n_posts:,d} Posts"
    ## Generate Distribution Plot
    all_counts_dist = pd.Series(all_counts).value_counts()
    ax[i].scatter(all_counts_dist.index, all_counts_dist.values, alpha=.6)
    ax[i].set_yscale("symlog")
    ax[i].set_xscale("symlog")
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)
    ax[i].set_xlabel("# Posts", fontsize=12, fontweight="bold")
    ax[i].set_title(summary_str, fontsize=14, fontweight="bold")
    ax[i].tick_params(labelsize=12)
ax[0].set_ylabel("# Users", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}post_distribution.png", dpi=300)
plt.close(fig)

## Temporal Distribution
for field, label in zip(["num_posts","num_users"], ["Posts","Users"]):
    fig, ax = plt.subplots(figsize=(10,5.8))
    comments[field].plot(ax=ax, color="C0", alpha=0.3, label="Comments")
    comments[field].rolling(7).mean().plot(ax=ax, color="C0", alpha=0.9, label="")
    submissions[field].plot(ax=ax, color="C1", alpha=0.3, label="Submissions")
    submissions[field].rolling(7).mean().plot(ax=ax, color="C1", alpha=0.9, label="")
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel(label, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}{field}_distribution_time.png", dpi=300)
    plt.close(fig)