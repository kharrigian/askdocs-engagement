
"""
Extract Metadata from Submissions. Some code adapted from
`demo_rules.py` (Alicia Nobles)
"""

## Script Meta Parameters
NUM_JOBS = 8

####################
### Imports
####################

## Standard Library
import re
import os
import sys
import json
import gzip
from glob import glob
from datetime import datetime
from functools import partial
from collections import Counter
from multiprocessing import Pool

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

####################
### Globals
####################

## Data
DATA_DIR = f"./data/raw/AskDocs/"

## Dates
MIN_DATE = "2017-01-01"; MIN_DATE = int(datetime.strptime(MIN_DATE, "%Y-%m-%d").timestamp())
MAX_DATE = "2021-01-01"; MAX_DATE = int(datetime.strptime(MAX_DATE, "%Y-%m-%d").timestamp())

## Accounts to Ignore
IGNORABLES = set([
    "AutoModerator",
    "[deleted]",
    "[removed]"
])

## Codebook
weekday_map = dict((w, i) for i, w in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
month_map = dict((m,i) for i, m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"]))
race_map = {"white":0,"black":1,"hispanic":2,"asian":3,"indian":4,"mixed":5,"middle_eastern":6,"pacifc_islander":7}
gender_map = {"male":0,"female":1}

####################
### Functions
####################

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


class PostStream(object):
    
    """

    """

    def __init__(self,
                 filenames,
                 kwargs={}):
        """

        """
        self.filenames = filenames
        self.kwargs = kwargs
        self._len = sum(list(map(lambda x: len(load_data(x,
                                                         filters=None,
                                                         exclude_ignorable_accounts=self.kwargs.get("exclude_ignoreable_accounts",True))), filenames)))
    
    def __len__(self):
        """

        """
        return self._len
    
    def __iter__(self):
        """

        """
        for file in self.filenames:
            file_data = load_data(file, **self.kwargs)
            for post in file_data:
                yield post

def clean_text(post,
               demo,
               text_fields=["title","selftext"]):
    """
    Args:
        post (dict): Post to process
        demo (str): e.g. "age","gender","race"
    """
    ## Clean up text - join title/body, remove new lines, strip punctuation
    text = list(filter(lambda i: i is not None, [post.get(tf,"") for tf in text_fields]))
    text = " ".join(text)
    text = text.replace('\n', ' ')
    text = text.lower()
    if demo == 'age':
        # including punc that is common for temp and height 99.0 or 5'11''
        exclude = set('!#$&\()*+,-/:;<=>?@[\\]^_`{|}~')
    else:
        exclude = set('!"#%$&\()*+,-./:;<=>?@[\\]^_`{|}~')
        apostrophe = set("'")
        apostrophe.add(chr(8217))
        apostrophe.add(chr(8216))
        for char in apostrophe:
            text = text.replace(char, '')
    for char in exclude:
        text = text.replace(char, ' ')
    text = re.sub('\s+', ' ', text).strip()
    return text

def _check_match(post,
                 pattern,
                 demo):
    """
    Clean text and check match against regular expression

    Args:
        post (dict)
        pattern (compiled regex)
        demo (str)
    
    Returns:
        match (str or None): If a match, return link_id for post. Otherwise None.
    """
    post_text = clean_text(post, demo)
    match = re.search(pattern, post_text)
    if match is not None:
        match = post.get("id")
    return match

def check_automod(posts):
    """
    Labels whether a post meets the automod standard as indicated by askdocs
    
    Args:
        posts (list of dict)
    
    Returns:
        automod_met (list): Acceptable IDs
    """
    ## Patterns that askdocs automod checks for
    automod_pat = ["age", "weight", "height", "male", "female", "lbs", "lbs.", "lb", "pounds", "years old", "year old", "y/o", "yr/old", "yrs", "yrs old", "cm", "kg", "weigh"]
    reg_pat = re.compile(r'\b(?:%s)\b' % '|'.join(automod_pat))
    ## Process
    mp = Pool(NUM_JOBS)
    helper = partial(_check_match, pattern=reg_pat, demo="automod")
    acceptable = list(tqdm(mp.imap_unordered(helper, posts), total=len(posts), file=sys.stdout, desc="Automod Verification"))
    acceptable = list(filter(lambda x: x is not None, acceptable))
    _ = mp.close()
    ## Sort and Return
    acceptable = sorted(acceptable)
    return acceptable

## Gender Extraction
def _find_gender(post, text_fields):
    """

    """
    ## Clean Text
    post_text = clean_text(post, 'gender', text_fields)
    ## Check for Matches
    matches = []
    if re.search(r'\bmale\b', post_text): # male
        matches.append('male')
    if re.search(r'\bm\b', post_text): # M
        matches.append('male')
    if re.search(r'\b[0-9]{1,2}m\b', post_text):# 24M
        matches.append('male')
    if re.search(r'\bman\b', post_text): # man
        matches.append('male')
    if re.search(r'\bboy\b', post_text):  # boy
        matches.append('male')
    if re.search(r'\bfemale\b', post_text): # female
        matches.append('female')
    if re.search(r'\bf\b', post_text): # F
        matches.append('female')
    if re.search(r'\b[0-9]{1,2}f\b', post_text): # 24F
        matches.append('female')
    if re.search(r'\bwoman\b', post_text): # woman
        matches.append('female')
    if re.search(r'\bgirl\b', post_text): # girl
        matches.append('female')
    if re.search(r'\b(f)(\d{1,2})', post_text) is not None: #F23
        matches.append('female')
    if re.search(r'\b(m)(\d{1,2})', post_text) is not None: #M23
        matches.append('male')
    ## Isolate Unique
    matches = set(matches)
    if len(matches) == 0 or len(matches) > 1:
        return post.get("id"), "unknown"
    return post.get("id"), matches.pop()

## Age Identification
def _find_age(post, text_fields):
    """

    """
    ## Clean Text and Get Post ID
    post_text = clean_text(post, 'age', text_fields)
    post_id = post.get("id")
    ## Cache for Matches
    patt_match = []
    ## Search for patterns and append the age
    match = re.search(r'i am\s([0-9]{1,2})\b(?! lb| kg| pound| kilo| percent|%|.| cm| meter)', post_text) # i am 24
    if match != None:
        patt_match.append(match.group(1))
    match = re.search(r"i'm\s([0-9]{1,2})\b(?! lb| kg| pound| kilo| percent|%|.| cm| meter)", post_text) # i'm 24
    if match != None:
        patt_match.append(match.group(1))
    match = re.search(r'\b([0-9]{1,2})\s(?:years old|year old|yrs old|yr old|yo|y o)', post_text) # 24 years old, 24 y/o, 24 yrs old
    if match != None:
        patt_match.append(match.group(1))
    match = re.search(r'\b([0-9]{1,2})\s(?:months old|month old|mos old|mo old)', post_text) # babies (months)
    if match != None:
        months = int(match.group(1))
        baby_age = months/12
        patt_match.append(baby_age)
    match = re.search(r'\b([0-9]{1,2})\s(?:days old|day old)', post_text) # babies (days)
    if match != None:
        days = int(match.group(1))
        baby_age = days/365
        patt_match.append(baby_age)
    match = re.search(r'\b([0-9]{1,2})\s(?:weeks old|week old|wk old|wks old)', post_text) # babies (weeks)
    if match != None:
        weeks = int(match.group(1))
        baby_age = weeks/52
        patt_match.append(baby_age)
    match = re.search(r'\b([0-9]{1,2})(?:m\b|f\b)', post_text) # 24M/24F
    if match != None:
        patt_match.append(match.group(1))
    match = re.search(r'\b(?:m|f)([0-9]{1,2})', post_text) # M24/F24
    if match != None:
        patt_match.append(match.group(1))
    match = re.search(r'age\s{1,2}([0-9]{1,2})\b', post_text) # age: 24, age 24
    if match != None:
        patt_match.append(match.group(1))
    ## If we haven't found an age using the above rules, we'll match to a common pattern (but less so than above)
    if len(patt_match) == 0:
        ## Common pattern male 24 or 24 male
        match1 = re.search(r'\b([0-9]{1,2})\s(?:\bmale|female)', post_text) # 24 male/female
        match2 = re.search(r'\b(?:male|female)\s([0-9]{1,2})', post_text) # male/female 24
        match3 = re.search(r'\b([0-9]{1,2})\s\b(?:m|f)\b', post_text) # 24 m/f
        match4 = re.search(r'\b(?:m|f)\s([0-9]{1,2})', post_text) # m/f 24
        if match1 != None:
            patt_match.append(match1.group(1))
        elif match2 != None:
            patt_match.append(match2.group(1))
        elif match3 != None:
            patt_match.append(match3.group(1))
        elif match4 != None:
            patt_match.append(match4.group(1))
    ## Omit common errors
    typical_errors = ('0','95','96','97','98','99')
    patt_match = list(map(str,patt_match))
    patt_match = set(i for i in patt_match if i not in typical_errors and i.isdigit())
    ## Get Match
    if len(patt_match) == 0 or len(patt_match) > 1:
        return post_id, "unknown"
    else:
        return post_id, patt_match.pop()

def _find_race(post, text_fields):
    """

    """
    ## Clean Text and Get ID
    post_text = clean_text(post, 'race')
    post_id = post.get("id")
    ## Find Matches
    patt_match = []
    if re.search(r'\b(?:white|caucasian)\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool| mold)', post_text):
        patt_match.append('white')
    if re.search(r'\b(?:african|black)\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool| mold)', post_text):
        patt_match.append('black')
    if re.search(r'\b(?:hispanic|latino|latina)\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool)', post_text):
        patt_match.append('hispanic')
    if re.search(r'\b(?:asian|chinese|japenese|korean|vietnamese)\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool)', post_text):
        patt_match.append('asian')
    if re.search(r'\bindian\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool)', post_text):
        patt_match.append('indian')
    if re.search(r'\bmixed\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool)', post_text):
        patt_match.append('mixed')
    if re.search(r'\bmiddle eastern\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool)', post_text):
        patt_match.append('middle_eastern')
    if re.search(r'\bpacific islander\b(?! bump| skin| spot| area| sore| hair| nail| fingernail| restaurant| food| teeth| tooth| streak| mark | syndrome| fluff| stuff| fluid| pus| bruise| stool)', post_text):
        patt_match.append('pacific_islander')
    ## Return
    patt_match = set(patt_match)
    if len(patt_match) == 0 or len(patt_match) > 1:
        return post_id, "unknown"
    else:
        return post_id, patt_match.pop()

## Labeler
def label_demo(posts,
               finder,
               demo,
               text_fields):
    """
    Labels demographics of poster using regular expressions
    """
    ## Helper
    finder_h = partial(finder, text_fields=text_fields)
    ## Search for Matches
    mp = Pool(NUM_JOBS)
    labels = dict(tqdm(mp.imap_unordered(finder_h, posts), total=len(posts), desc=f"{demo.title()} Labeling", file=sys.stdout))
    _ = mp.close()
    return labels

def _get_metadata(post):
    """

    """
    ## Extract Text
    post_title = clean_text(post, "", ["title"])
    post_selftext = clean_text(post, "", ["selftext"])
    ## Length
    len_title = len(post_title.split())
    len_selftext = len(post_selftext.split())
    ## Meta
    post_created_dt = datetime.fromtimestamp(post.get("created_utc"))
    meta = {
        "author":post.get("author"),
        "created_utc":post.get("created_utc"),
        "created_utc_hour":post_created_dt.hour,
        "created_utc_weekday":datetime.strftime(post_created_dt, "%A"),
        "created_utc_month":datetime.strftime(post_created_dt, "%B"),
        "created_utc_year":post_created_dt.year,
        "title_length":len_title,
        "selftext_length":len_selftext
    }
    return post.get("id"), meta

## Post Metadata
def get_metadata(posts):
    """

    """
    mp = Pool(NUM_JOBS)
    metadata = dict(tqdm(mp.imap_unordered(_get_metadata, posts), total=len(posts), desc="Metadata Extractor"))
    _ = mp.close()
    return metadata

def main():
    """

    """
    ## Get Filenames and Initialize Post Strem
    submission_filenames = sorted(glob(f"{DATA_DIR}raw/AskDocs/submissions/*.json.gz"))
    submission_stream = PostStream(submission_filenames)
    ## Automod Formatting
    automod_format = check_automod(submission_stream)
    ## Demographic Labeling
    demographic_labels = {"gender":{},"age":{},"race":{}}
    for field, text_fields in zip(["title","selftext","all"],[["title"],["seltext"],["title","selftext"]]):
        print(f"Getting Labels in {field}")
        for func, demo in zip([_find_gender,_find_age,_find_race],["gender","age","race"]):
            demographic_labels[demo][field] = label_demo(submission_stream, func, demo, text_fields)
    ## Post Metadata
    post_metadata = get_metadata(submission_stream)
    ## Merge Metadata
    metadata_df = pd.DataFrame(post_metadata).T
    metadata_df["is_automod_format"] = metadata_df.index.isin(automod_format)
    for demo, demo_dict in demographic_labels.items():
        for field, field_labels in demo_dict.items():
            metadata_df[f"{demo}_{field}"] = metadata_df.index.map(lambda i: field_labels.get(i))
    for demo in demographic_labels.keys():
        for field in ["title","selftext"]:
            metadata_df.loc[metadata_df[f"{demo}_all"]=="unknown", f"{demo}_{field}"] = "unknown"
    metadata_df = metadata_df.replace("unknown",np.nan)
    ## Recode Metadata
    metadata_df["created_utc_weekday"] = metadata_df["created_utc_weekday"].map(lambda i: weekday_map.get(i))
    metadata_df["created_utc_month"] = metadata_df["created_utc_month"].map(lambda i: month_map.get(i))
    for field in ["title","selftext","all"]:
        for demo, demo_codes in zip(["gender","race"],[gender_map, race_map]):
            metadata_df[f"{demo}_{field}"] = metadata_df[f"{demo}_{field}"].map(lambda i: demo_codes.get(i) if not pd.isna(i) else np.nan)
    demo_cols = [c for c in metadata_df.columns if any(c.startswith(d) for d in ["gender","age","race"])]
    metadata_df[demo_cols] = metadata_df[demo_cols].fillna(-1).astype(int)
    ## Filter by Date
    metadata_df = metadata_df.loc[metadata_df["created_utc"].map(lambda i: i >= MIN_DATE and i < MAX_DATE)]
    metadata_df = metadata_df.copy()
    metadata_df.index.name = "id"
    ## Dump
    metadata_df.to_csv(f"{DATA_DIR}processed/submission_metadata.csv",index=True)
    print("Script Complete")

###################
### Execution
###################

if __name__ == "__main__":
    _ = main()