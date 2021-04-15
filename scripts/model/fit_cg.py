
"""

"""

## Plot Directory
PLOT_DIR = "./plots/model/cg/received_any_response/"

## Dependent Variable
DEPENDENT_VARIABLE = "received_any_response"

## Dependent Variable Parameters
LOGTRANSFORM = True

## Independent Variable Parameters
DROP_NON_AUTOMOD = True
DROP_DEMO_LOC = True
DROP_DELETED = True
TOPIC_THRESH = 0.001
TOPIC_THRESH = 0.025
TOPIC_REPRESENTATION = "continuous" ## "continuous", "binary", "max"

## Temporal Linkages
TEMPORAL_WINDOW_BEFORE = 60 * 10 # seconds
TEMPORAL_WINDOW_AFTER = 60 * 5 # seconds
TEMPORAL_WINDOW_BEFORE = 2 # samples
TEMPORAL_WINDOW_AFTER = 1 # samples
MAX_TEMPORAL_WINDOW_BEFORE = 10
MAX_TEMPORAL_WINDOW_AFTER = 5
IS_FIXED_WINDOW = True  ## If true, use sample-based window instead of temporal

###################
### Imports
###################

## Standard Libary
import os
import sys
from datetime import datetime
from itertools import combinations

## External
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ananke import graphs, models, estimation

###################
### Globals
###################

## Data Directory
DATA_DIR = "./data/"

## Check Arguments
if TOPIC_REPRESENTATION not in set(["continuous","binary","max"]):
    raise ValueError("Topic representation must be one of continuous, binary, or max")

## Variable Types
IV_VARIABLE_TYPES = {'created_utc_hour': "categorical",
                     'created_utc_weekday': "categorical",
                     'created_utc_month': "categorical",
                     'created_utc_year': "categorical",
                     'title_length': "ordinal",
                     'selftext_length': "ordinal",
                     'is_automod_format': "binary",
                     'gender_title': "categorical",
                     'gender_selftext': "categorical",
                     'gender_all': "categorical",
                     'age_title': "categorical",
                     'age_selftext': "categorical",
                     'age_all': "categorical",
                     'race_title': "categorical",
                     'race_selftext': "categorical",
                     'race_all': "categorical",
                     'n_prior_submissions': "ordinal"}
DV_VARIABLE_TYPES = {'n_response': "ordinal",
                     'n_response_deleted': "ordinal",
                     'n_unique_responders': "ordinal",
                     'received_physician_response': "binary",
                     'received_physician_in_training_response': "binary",
                     'received_non_physician_provider_response': "binary",
                     'received_non_physician_provider_in_training_response': "binary",
                     'received_any_provider_response': "binary",
                     'received_any_response': "binary",
                     'time_to_first_response': "continuous",
                     'time_to_physician_first_response': "continuous",
                     'time_to_non_physician_provider_first_response': "continuous",
                     'time_to_physician_in_training_first_response': "continuous",
                     'time_to_non_physician_provider_in_training_first_response': "continuous",
                     'time_to_any_provider_response_first_response': "continuous"}

## Age Bins
age_bins = [0, 13, 20, 35, 50, 100]

## Codebook
weekday_map = dict((i, w) for i, w in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
month_map = dict((i,m) for i, m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"]))
race_map = dict((i,r) for i, r in enumerate(["white","black","hispanic","asian","indian","mixed","middle_eastern","pacific_islander"]))
age_map = dict((i, f"{l}_{u}") for i, (l,u) in enumerate(zip(age_bins[:-1],age_bins[1:])))
gender_map = {0:"male",1:"female"}
topic_map = dict()

## Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

###################
### Helpers
###################

def classify_covariate(covariate, topic_variables):
    """

    """
    if covariate.startswith("race"):
        c_type = "demo_race"
    elif covariate.startswith("gender"):
        c_type = "demo_gender"
    elif covariate.startswith("age"):
        c_type = "demo_age"
    elif covariate in topic_variables:
        c_type = "topic"
    elif covariate.endswith("_length") or covariate.startswith("is_automod_format"):
        c_type = "description"
    elif any(covariate.startswith(char) for char in ["created_utc_hour","created_utc_weekday"]):
        c_type = "temporal_alpha"
    elif covariate.startswith("created_utc_year") or covariate.startswith("created_utc_month"):
        c_type = "temporal_beta"
    elif covariate == "n_prior_submissions":
        c_type = "participation"
    elif covariate == "response":
        c_type = "response"
    else:
        raise ValueError("Covariate type not recognized")
    return c_type

def bin_data(value, bins):
    """

    """
    if value == -1:
        return -1
    for i, (lower, upper) in enumerate(zip(bins[:-1],bins[1:])):
        if value >= lower and value < upper:
            return i
    return None


def format_data_for_cg(X,
                       Y,
                       tau,
                       topic_columns):
    """

    """
    ## Data Formatting
    data = pd.merge(X, Y[[DEPENDENT_VARIABLE]], left_index=True, right_index=True)
    if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
        data[DEPENDENT_VARIABLE] = data[DEPENDENT_VARIABLE].astype(int)
    ## One Hot Encoding
    for col in list(data.columns):
        if col == DEPENDENT_VARIABLE:
            continue
        if IV_VARIABLE_TYPES[col] in ["categorical","binary"] and len(set(data[col])) > 1:
            data[col] = data[col].astype(int)
            data, _ = convert_to_one_hot(data, col, keep_all=data[col].min()==-1)
    ## Data Scaling
    for covariate in data.columns.tolist():
        if covariate == DEPENDENT_VARIABLE or covariate[-1].isdigit():
            continue
        data[covariate] = (data[covariate] - data[covariate].mean()) / data[covariate].std()
    ## Clean Topics for Patsy Support
    clean_str = lambda t: t.replace("(","").replace(")","").replace(" ","_").lower().replace("-","_").replace(",","")
    topic_cols_clean = [clean_str(t) for t in topic_columns]
    data = data.rename(columns=dict(zip(topic_columns, topic_cols_clean)))
    ## Clean One-hots for Patsy Support
    for field in ["selftext","title","all"]:
        for demo in ["gender","race","age"]:
            demo_field_cols = [c for c in data.columns if c.startswith(f"{demo}_{field}")]
            demo_labels = [f"{demo}_{field}_{name}" for name in format_feature_names(demo_field_cols, demo)]
            if len(demo_labels) != 0:
                data = data.rename(columns=dict(zip(demo_field_cols, demo_labels)))
    ## Unique Variables
    covariates = data.columns.tolist()
    covariates = [c for c in covariates if c != DEPENDENT_VARIABLE]
    ## CG Alignment and Window
    post_ids = data.index.tolist()
    before_n = max([len(t["before"]) for t in tau])
    after_n = max([len(t["after"]) for t in tau])
    ## Add Null Row
    data = pd.concat([data, pd.DataFrame(index=[None])])
    ## Isolate Appropriate Window Data
    window_data = []
    missing_inds = set()
    for rn, rn_range in zip([before_n,after_n],["before","after"]):
        print(rn_range)
        for n in range(rn):
            print(n)
            n_ids = list(map(lambda t: t[rn_range][n] if len(t[rn_range])> n else None,tau))
            missing_inds.update([i for i, n in enumerate(n_ids) if n is None])
            n_data = data.loc[n_ids].copy()
            n_data = n_data.rename(columns=dict((c, f"{c}_{rn_range}_{n+1}") for c in data.columns))
            n_data = n_data.reset_index(drop=True)
            window_data.append(n_data)
    ## Merge Data
    window_data = pd.concat(window_data, axis=1)
    data = data.iloc[:-1]
    data = data.reset_index(drop=True)
    data = pd.merge(data,
                    window_data,
                    left_index=True,
                    right_index=True)
    nn_inds = [i for i in range(data.shape[0]) if i not in missing_inds]
    data = data.iloc[nn_inds]
    return data, covariates, topic_cols_clean

def construct_chain_graph(covariates,
                          dependent_variable,
                          topic_variables,
                          tau):
    """

    Demos (Age, Gender, Race) -> Topic, Temporal, Description Quality, Response
    Participation -> Description, Response
    Topic -> Description Quality, Response
    Description Quality -> Response
    Temporal -> Response
    """
    ## Chain Range
    before_n = max([len(t["before"]) for t in tau])
    after_n = max([len(t["after"]) for t in tau])
    ## Classify Covariates
    labels = covariates + ["response"]
    labels2ind = dict(zip(labels, range(len(labels))))
    label_types = [classify_covariate(l, topic_variables) for l in labels]
    label_types_map = {}
    for unique_label_type in set(label_types):
        label_types_map[unique_label_type] = set([label for label, label_type in zip(labels, label_types) if label_type == unique_label_type])
    ## Mappings
    DAG_map = {
        "demo_age":["topic","temporal_alpha","description","response"],
        "demo_gender":["topic","temporal_alpha","description","response"],
        "demo_race":["topic","temporal_alpha","description","response"],
        "participation":["description","response"],
        "topic":["description","response"],
        "description":["response"],
        "temporal_alpha":["response"],
        "temporal_beta":["response"],
        "response":[]
    }
    ## Construct Graph
    vertices = set()
    directed_edges = []
    undirected_edges = []
    inds = [""] + \
           [f"_before_{b+1}" for b in range(before_n)] + \
           [f"_after_{a+1}" for a in range(after_n)]
    for ind in inds:
        for c1, c1name in enumerate(labels):
            c1_type = label_types[c1]
            for c2_type in DAG_map[c1_type]:
                c2_values = label_types_map[c2_type]
                for c2v in c2_values:
                    c1_ = c1name if c1name != "response" else dependent_variable
                    c2_ = c2v if c2v != "response" else dependent_variable
                    directed_edges.append((f"{c1_}{ind}",f"{c2_}{ind}"))
                    vertices.update(directed_edges[-1])
    for ind1 in inds:
        for ind2 in inds:
            if ind1 == ind2:
                continue
            for topic_var in topic_variables:
                directed_edges.append((f"{topic_var}{ind1}",f"{dependent_variable}{ind2}"))
    for ind1, ind2 in combinations(inds, 2):
        undirected_edges.append((f"{dependent_variable}{ind1}",f"{dependent_variable}{ind2}"))
    vertices = sorted(vertices)
    return vertices, directed_edges, undirected_edges

def convert_to_one_hot(X,
                       covariate,
                       keep_all=False):
    """

    """
    targets = sorted(X[covariate].unique())
    if not keep_all:
        targets = targets[1:]
    X_one_hot = [(X[covariate]==target).astype(int).to_frame(f"{covariate}_{target}") for target in targets]
    X_one_hot = pd.concat(X_one_hot, axis=1)
    one_hot_columns = X_one_hot.columns.tolist()
    X = X.drop(covariate,axis=1)
    X = pd.merge(X, X_one_hot, left_index=True, right_index=True)
    return X, one_hot_columns

def _format_feature_names(feature_names, mapping):
    """

    """
    feature_names_formatted = []
    for f in feature_names:
        find = int(f.split("_")[-1])
        feature_names_formatted.append(mapping.get(find,"unknown"))
    return feature_names_formatted

def format_feature_names(features, variable_type):
    """

    """
    if variable_type == "gender":
        feature_labels = _format_feature_names(features, gender_map)
    elif variable_type == "race":
        feature_labels = _format_feature_names(features, race_map)
    elif variable_type == "age":
        feature_labels = _format_feature_names(features, age_map)
    elif variable_type == "created_utc_weekday":
        feature_labels = _format_feature_names(features, weekday_map)
    elif variable_type == "created_utc_month":
        feature_labels = _format_feature_names(features, month_map)
    elif variable_type == "topic" and TOPIC_REPRESENTATION == "max":
        feature_labels = _format_feature_names(features, topic_map)
    else:
        feature_labels = features
    return feature_labels

def construct_temporal_adjaceny(X,
                                before_window=None,
                                after_window=None,
                                max_before_window=None,
                                max_after_window=None,
                                is_fixed=False,
                                cache_ids=True):
    """
    Args:
        X (pandas DataFrame)
        before_window (int or None): Either number of samples (is_fixed=True) or time in seconds
        after_window (int or None): Either number of samples (is_fixed=True) or time in seconds
        max_before_window (int or None): If specified, max samples before
        max_after_window (int or None): If specified, max samples after
        is_fixed (bool): If True, windows based on samples instead of time
        cache_ids (bool): If True, cache post IDs instead of indices
    Returns:
        adjacencies (list): Posts adjacent, aligned with index of X
    """
    ## Check Arguments
    if before_window is None and after_window is None:
        raise ValueError("Need to specify a window size (either upper, lower, or both)")
    elif before_window is None:
        before_window = 0
    elif after_window is None:
        after_window = 0
    if max_before_window is None:
        max_before_window = np.inf
    if max_after_window is None:
        max_after_window = np.inf
    ## Sort Data
    created_utc = X.sort_values("created_utc",ascending=False)["created_utc"]
    post_ids = created_utc.index.tolist()
    created_utc = created_utc.values
    ## Update Windows
    if is_fixed:
        max_before_window = before_window
        max_after_window = after_window
        before_window = np.inf
        after_window = np.inf
    ## Construct Adjacencies
    n = created_utc.shape[0]
    i = 0 ## Current index
    j = 1 ## Before index
    k = 0 ## After index
    adjacencies = {}
    print("Identifying Temporal Adjacencies")
    while i < n:
        if (i + 1) % (n // 10) == 0:
            print(f"Adjacency {i+1}/{n} Complete.")
        i_adj_bef = []
        i_adj_aft = []
        if cache_ids:
            i_adj_bef.extend([(z, post_ids[z]) for z in range(i+1, j)])
        else:
            i_adj_bef.extend([(z,z) for z in list(range(i+1, j))])
        while (j < n) and (created_utc[i] - created_utc[j] < before_window) and (j - i <= max_before_window):
            if cache_ids:
                i_adj_bef.append((j,post_ids[j]))
            else:
                i_adj_bef.append((j,j))
            j += 1
        while (k < i) and ((created_utc[k] - created_utc[i] >= after_window) or (i - k > max_after_window)):
            k += 1
        if cache_ids:
            i_adj_aft.extend([(z,post_ids[z]) for z in range(k, i)])
        else:
            i_adj_aft.extend([(z,z) for z in list(range(k, i))])
        ## Sorting (Closest to Post to Further Away)
        i_adj_bef = [z[1] for z in sorted(i_adj_bef, key=lambda z: z[0])]
        i_adj_aft = [z[1] for z in sorted(i_adj_aft, key=lambda z: z[0], reverse=True)]
        adjacencies[post_ids[i]] = {"before":i_adj_bef, "after":i_adj_aft}
        i += 1
    ## Sort Relative to Original Input
    adjacencies = [adjacencies[post_id] for post_id in X.index.tolist()]
    return adjacencies

###################
### Load Data
###################

print("Loading Data")

## Load Variables
X = pd.read_csv(f"{DATA_DIR}/processed/independent.csv",index_col=0)
Y = pd.read_csv(f"{DATA_DIR}/processed/dependent.csv",index_col=0)

## Drop Deleted Posts (Contain No Information)
if DROP_DELETED:
    deleted_mask = ~X["post_deleted"]
    X = X[deleted_mask]
    Y = Y[deleted_mask]
X = X.drop("post_deleted",axis=1)

## Ignore Posts Removed by Automoderator
if DROP_NON_AUTOMOD:
    automod_mask = np.logical_and(~Y["post_removed_missing_detail"].fillna(False),
                                  ~Y["post_removed_emergency"].fillna(False))
    X = X.loc[automod_mask]
    Y = Y.loc[automod_mask]
Y = Y.drop(["post_removed_missing_detail","post_removed_emergency"],axis=1)

## Demo Locations
if DROP_DEMO_LOC:
    demo_loc_cols = [c for c in X.columns if any(c.startswith(char) and not c.endswith("_all") for char in ["race","age","gender"])]
    X = X.drop(demo_loc_cols,axis=1)
    for dlc in demo_loc_cols:
        _ = IV_VARIABLE_TYPES.pop(dlc, None)

## Age Bins
for field in ["selftext","title","all"]:
    if f"age_{field}" not in X.columns:
        continue
    X[f"age_{field}"] = X[f"age_{field}"].map(lambda a: bin_data(a, age_bins))

## Drop Outliers / Anomalies (e.g. Negative Values)
if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "continuous":
    cont_mask = np.logical_and(~Y[DEPENDENT_VARIABLE].isnull(), Y[DEPENDENT_VARIABLE] > 0)
    X = X[cont_mask]
    Y = Y[cont_mask]

######################
### Timing Distribution
######################

## Variables Measuring Response Time
timing_variables = [i for i in Y.columns if i.startswith("time_to")]

## Plot Timings
time_range = np.arange(0,361,1)
fig, ax = plt.subplots(figsize=(10,5.8))
for tv in timing_variables:
    tv_d = Y[tv].dropna()
    tv_resp = pd.Series([(tv_d < i).value_counts(normalize=True).get(True,0)*100 for i in time_range],
                        index=time_range)
    ax.plot(time_range,
            tv_resp,
            alpha=0.5,
            label=tv.replace("_"," ").title())
ax.set_xlabel("Threshold (Minutes)", fontweight="bold", fontsize=16)
ax.set_ylabel("Proportion of Users", fontweight="bold", fontsize=16)
ax.legend(loc="lower right")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=14)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig(f"./plots/metadata_distributions/time_to_response_curves.png", dpi=200)
plt.close(fig)

######################
### Modeling Prep
######################

## Log Transformation
if LOGTRANSFORM and DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "continuous":
    Y[DEPENDENT_VARIABLE] = np.log(Y[DEPENDENT_VARIABLE])

## Topic Columns
topic_columns = X.columns[[i for i, c in enumerate(X.columns) if c == "max_topic"][0]+1:].tolist()
topic_columns_meets_thresh = ((X[topic_columns]!=0).mean(axis=0) > TOPIC_THRESH)
X.drop(topic_columns_meets_thresh.loc[~topic_columns_meets_thresh].index.tolist(),
       axis=1,
       inplace=True)
topic_columns = topic_columns_meets_thresh.loc[topic_columns_meets_thresh].index.tolist()
topic_map = dict((i, topic) for i, topic in enumerate(topic_columns))
topic_map_r = dict((y,x) for x, y in topic_map.items())
X["max_topic"] = X["max_topic"].map(lambda i: topic_map_r.get(i,-1))

## Topic Formatting
if TOPIC_REPRESENTATION == "continuous":
    X = X.drop("max_topic",axis=1)
    for topic_col in topic_columns:
        IV_VARIABLE_TYPES[topic_col] = "continuous"
elif TOPIC_REPRESENTATION == "binary":
    X[topic_columns] = (X[topic_columns] > 0).astype(int)
    X = X.drop("max_topic",axis=1)
    for topic_col in topic_columns:
        IV_VARIABLE_TYPES[topic_col] = "binary"
elif TOPIC_REPRESENTATION == "max":
    X = X.drop(topic_columns, axis=1)
    topic_columns = ["max_topic"]
    IV_VARIABLE_TYPES["max_topic"] = "binary"
    X["max_topic"] = X["max_topic"].astype(int)
else:
    raise ValueError("Unexpected topic representation")

## Temporal Adjacency List (For Chain Graph)
tau = construct_temporal_adjaceny(X,
                                  before_window=TEMPORAL_WINDOW_BEFORE,
                                  after_window=TEMPORAL_WINDOW_AFTER,
                                  max_before_window=MAX_TEMPORAL_WINDOW_BEFORE,
                                  max_after_window=MAX_TEMPORAL_WINDOW_AFTER,
                                  is_fixed=IS_FIXED_WINDOW,
                                  cache_ids=True)

## Format Covariates
X = X.drop("created_utc",axis=1)
for column in X.columns:
    if column == "max_topic" or column in topic_columns:
        continue
    X[column] = X[column].astype(int)

## Format Targets
Y["n_response_deleted"] = Y["n_response_deleted"].fillna(0)

## Isolate Variables
covariates = X.columns.tolist()

######################
### Graphical Model (Simple Visualization)
######################

"""
T1: Topic 1
T2: Topic 2
D: Demographics
C: Temporal Metadata
Q: Description Quality
Y: Outcome
"""

unique_vars = ["T1","T2","D","C","Q","Y"] 
cg_vertices = unique_vars + \
              [f"{v}_A1" for v in unique_vars] + \
              [f"{v}_B1" for v in unique_vars]
cg_di_edges = []
cg_ud_edges = []
for sample in ["","_A1","_B1"]:
    cg_di_edges.append((f"D{sample}",f"T1{sample}"))
    cg_di_edges.append((f"D{sample}",f"T2{sample}"))
    cg_di_edges.append((f"D{sample}",f"Q{sample}"))
    cg_di_edges.append((f"D{sample}",f"Y{sample}"))
    cg_di_edges.append((f"D{sample}",f"C{sample}"))
    cg_di_edges.append((f"C{sample}",f"Y{sample}"))
    cg_di_edges.append((f"T1{sample}",f"Q{sample}"))
    cg_di_edges.append((f"T1{sample}",f"Y{sample}"))
    cg_di_edges.append((f"T2{sample}",f"Q{sample}"))
    cg_di_edges.append((f"T2{sample}",f"Y{sample}"))
    cg_di_edges.append((f"Q{sample}",f"Y{sample}"))
    for sample2 in ["","_A1","_B1"]:
        if sample == sample2:
            continue
        cg_di_edges.append((f"T1{sample}",f"Y{sample2}"))
        cg_di_edges.append((f"T2{sample}",f"Y{sample2}"))
for s1, s2 in list(combinations(["","_A1","_B1"],2)):
    cg_ud_edges.append((f"Y{s1}",f"Y{s2}"))

## Initialize Graph
cg_vis = graphs.CG(vertices=cg_vertices, di_edges=cg_di_edges, ud_edges=cg_ud_edges)
_ = cg_vis.draw().render(f"{PLOT_DIR}/cg_simple.gv", view=False)

######################
### Graphical Model 
######################

## Format Data for CG
data, covariates, topic_columns_formatted = format_data_for_cg(X=X,
                                                               Y=Y,
                                                               tau=tau,
                                                               topic_columns=topic_columns)

## Create Graph
vertices, directed_edges, undirected_edges = construct_chain_graph(covariates=covariates,
                                                                   dependent_variable=DEPENDENT_VARIABLE,
                                                                   topic_variables=topic_columns_formatted,
                                                                   tau=tau)
cg = graphs.CG(vertices=vertices,
               di_edges=directed_edges,
               ud_edges=undirected_edges)

## 
