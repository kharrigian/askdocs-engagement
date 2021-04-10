
"""
Analyze the merged metadata that will be used for modeling
"""

## Meta Parameters
APPLY_DATE_FILTER = False
MIN_DATE = "2019-01-01"
MAX_DATE = "2021-01-01"

## Plot Directory
PLOT_DIR = "./plots/metadata_distributions/"

####################
### Imports
####################

## Standard Library
import os
import sys
from datetime import datetime
from itertools import product
from collections import Counter

## External Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from pandas.plotting import register_matplotlib_converters
from sklearn import metrics

#####################
### Globals
#####################

## Data Directory
DATA_DIR = "./data/"

## Dates
MIN_DATE = int(datetime.strptime(MIN_DATE, "%Y-%m-%d").timestamp())
MAX_DATE = int(datetime.strptime(MAX_DATE, "%Y-%m-%d").timestamp())

## Register Converters
_ = register_matplotlib_converters()

## Codebook
weekday_map = dict((i, w) for i, w in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
month_map = dict((i, m) for i, m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"]))
race_map = {0: 'white', 1: 'black', 2: 'hispanic', 3: 'asian', 4: 'indian', 5: 'mixed', 6: 'middle_eastern', 7: 'pacifc_islander'}
gender_map = {0:"male",1:"female"}

## Plot Directories
topic_plot_dir = f"{PLOT_DIR}/topics_demographics/"
demo_plot_dir = f"{PLOT_DIR}/demographics/"
for demographic in ["gender_all","race_all","age_all"]:
    if not os.path.exists(f"{topic_plot_dir}/{demographic}/"):
        _ = os.makedirs(f"{topic_plot_dir}/{demographic}/")
if not os.path.exists(demo_plot_dir):
    _ = os.makedirs(demo_plot_dir)

#####################
### Helpers
#####################

def bootstrap(x,
              n_iter=10,
              alpha=0.05,
              aggfunc=np.nanmean,
              return_type=tuple):
    """

    """
    q = aggfunc(x)
    cache = np.zeros(n_iter)
    for n in range(n_iter):
        cache[n] = aggfunc(np.random.choice(x, replace=True, size=len(x)))
    q_ = np.percentile(q - cache, q=[alpha*100/2, 100-(alpha*100/2)])
    q_ = [q+q_[0], q, q+q_[1]]
    q_ = return_type(q_)
    return q_

def plot_outcome_distribution(data,
                              ivs,
                              dv,
                              variable_maps=None,
                              smoothing_window=7,
                              iv_counts=False,
                              time=True):
    """

    """
    ## Check Input
    if len(ivs) > 2:
        raise ValueError("Function only supports 2 independent variables")
    ## Isolate Relevant Data
    plot_df = data
    ## Aggregate
    grouper = pd.Grouper(key="created_utc_dt", freq=f"{smoothing_window}D")
    if iv_counts:
        if not time:
            plot_df_agg = plot_df.groupby(ivs).size().map(lambda i: (i,i,i)).to_frame("bootstrap")
        else:
            plot_df_agg = plot_df.groupby(ivs + [grouper])[dv].agg(lambda x: len(x) / smoothing_window).map(lambda i: (i,i,i)).to_frame("bootstrap")
    else:
        if not time:
            plot_df_agg = plot_df.groupby(ivs)[dv].agg(bootstrap).to_frame("bootstrap")
        else:
            plot_df_agg = plot_df.groupby(ivs + [grouper])[dv].agg(bootstrap).to_frame("bootstrap")
    ## Parse
    for q, qname in enumerate(["lower","median","upper"]):
        plot_df_agg[qname] = plot_df_agg["bootstrap"].map(lambda i: i[q])
    plot_df_agg = plot_df_agg.sort_index()
    ## Generate Plot
    linestyles = ["-","--","-.",":",(0, (5, 10)),(0, (3, 5, 1, 5, 1, 5))]
    fig, ax = plt.subplots(figsize=(12,5.8))
    ticklabels = []
    indices = list(product(*plot_df_agg.index.levels[:-1])) if time else plot_df_agg.index.tolist()
    for i, index in enumerate(indices):
        index_agg = plot_df_agg.loc[index]
        if not isinstance(index, tuple):
            index = [index]
        if not time:
            color = "C0"
        else:
            color = f"C{index[0]+1}"
        if variable_maps is not None:
            color_name = variable_maps[0].get(index[0],"unk.")
            line_name = ""
        else:
            color_name = None
            line_name = None
        line = "-"
        if len(index) > 1:
            line = linestyles[(index[1] + 1) % 6]
            if variable_maps is not None:
                line_name = variable_maps[1].get(index[1],"unk.")
        label_name = "{} {}".format(color_name if color_name is not None else "",
                                    line_name if line_name is not None else "").strip()
        if len(label_name) == 0:
            label_name = str(index)
        ticklabels.append(label_name)
        if time:
            ax.fill_between(index_agg.index,
                            index_agg["lower"],
                            index_agg["upper"],
                            alpha=0.1,
                            color=color)
            index_agg["median"].plot(ax=ax,
                                    color=color,
                                    linestyle=line,
                                    label=label_name)
        else:
            ax.bar([i],
                   [index_agg["median"]],
                    yerr=np.array([[index_agg["median"]-index_agg["lower"],
                                    index_agg["upper"]-index_agg["median"]]]).T,
                    color=color)
    ax.set_ylim(bottom=0)
    if time:
        ax.legend(loc="best", frameon=True)
    else:
        ax.set_xticks(range(i+1))
        ax.set_xticklabels(ticklabels,rotation=45,ha="right")
    if not iv_counts:
        ax.set_ylabel(dv.replace("_"," ").title(), fontweight="bold")
    else:
        ax.set_ylabel("Average Volume")
    if time:
        ax.set_xlabel("Date", fontweight="bold")
    else:
        ax.set_xlabel("Independent Variable", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax

#####################
### Data Loading
#####################

## Load Independent and Dependent Variables
X = pd.read_csv(f"{DATA_DIR}/processed/independent.csv",index_col=0)
Y = pd.read_csv(f"{DATA_DIR}/processed/dependent.csv",index_col=0)

## Topic Columns
topic_columns = X.columns[[i for i, c in enumerate(X.columns) if c == "max_topic"][0]+1:].tolist()
topic_map = dict((i, topic) for i, topic in enumerate(topic_columns))
topic_map_r = dict((y,x) for x, y in topic_map.items())
X["max_topic"] = X["max_topic"].map(lambda i: topic_map_r.get(i,-1))

## Date Filtering
if APPLY_DATE_FILTER:
    date_filter = np.logical_and(X["created_utc"] >= MIN_DATE, X["created_utc"] < MAX_DATE)
    X = X.loc[date_filter]
    Y = Y.loc[date_filter]

## Convert Datetime
X["created_utc_dt"] = X["created_utc"].map(datetime.fromtimestamp)

## Merge DataFrames
data_df = pd.merge(X, Y, left_index=True, right_index=True)

#####################
### Visualization
#####################

## Topic Distributions As a Function of Demographcis
## Do certain demographics discuss certain topics more frequently than others?
topics_data_df = data_df[["gender_all","race_all","age_all"]+topic_columns].copy()
topics_data_df[topic_columns] = (topics_data_df[topic_columns]>0)
for variable, variable_map in zip(["gender_all","race_all","age_all"],[[gender_map],[race_map],None]):
    print("Analyzing Topic Distributions as a Function of {}".format(variable))
    variable_dist = topics_data_df[variable].value_counts(normalize=True).drop(-1)
    for topic in tqdm(topic_columns, total=len(topic_columns), desc=f"{variable} Distributions", leave=False):
        ## Proportion of Posts from Group
        fig, ax = plot_outcome_distribution(topics_data_df,
                                            ivs=[variable],
                                            dv=topic,
                                            variable_maps=variable_map,
                                            iv_counts=False,
                                            time=False)
        fig.savefig(f"{topic_plot_dir}/{variable}/Proportion of Group {topic}.png", dpi=200)
        plt.close(fig)
        ## Proportion of Topic for Each Group
        topic_var_dist = topics_data_df.loc[topics_data_df[topic]][variable].value_counts(normalize=True).sort_index()
        if -1 in topic_var_dist:
            topic_var_dist = topic_var_dist.drop(-1)
        missing = variable_dist.index[~variable_dist.index.isin(topic_var_dist.index)]
        topic_var_dist = np.log((topic_var_dist / variable_dist).fillna(0) + 1e-5)
        topic_var_dist.loc[missing] = np.nan
        topic_var_dist.index = topic_var_dist.index.map(lambda i: variable_map[0].get(i, "unk.") if variable_map is not None else i)
        fig, ax = plt.subplots(figsize=(12,5.8))
        topic_var_dist.plot.barh(ax=ax, color="C0")
        ax.set_ylabel("Demographic",fontweight="bold")
        ax.set_xlabel("Proportion of Topic\n(Relative to Population)",fontweight="bold")
        ax.set_title(topic)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0,color="black",linestyle="-",alpha=0.5,zorder=-1)
        fig.tight_layout()
        fig.savefig(f"{topic_plot_dir}/{variable}/Proportion of Topic {topic}.png", dpi=200)
        plt.close(fig)

## What is the distribution of Topics?
topic_dist = data_df["max_topic"].value_counts().sort_values()
topic_dist.index = topic_dist.index.map(lambda i: topic_map.get(i, "Unknown"))
fig, ax = plt.subplots(figsize=(5.8,10))
topic_dist.drop("Unknown").plot.barh(ax=ax)
plt.tick_params(axis="y",labelsize=5)
plt.xscale("symlog")
plt.xlabel("# Submissions", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{topic_plot_dir}topic_distribution.png",dpi=200)
plt.close()

## Response rate as a function of topics
## Do certain topics have higher response rates than others?
topic_dist_rr = data_df.groupby(["max_topic"])["received_any_response"].agg(bootstrap).to_frame("bootstrap")
for i, name in enumerate(["lower","median","upper"]):
    topic_dist_rr[name] = topic_dist_rr["bootstrap"].map(lambda j: j[i])
topic_dist_rr.index = topic_dist_rr.index.map(lambda i: topic_map.get(i, "Unknown"))
topic_dist_rr = topic_dist_rr.loc[topic_dist.loc[topic_dist > 5].index].sort_values("median")
topic_dist_rr = topic_dist_rr.drop("Unknown")
fig, ax = plt.subplots(figsize=(5.8,10))
ax.barh(list(range(topic_dist_rr.shape[0])),
        topic_dist_rr["median"].values,
        xerr=np.array([(topic_dist_rr["median"]-topic_dist_rr["lower"]).values,
                       (topic_dist_rr["upper"]-topic_dist_rr["median"]).values]))
plt.yticks(list(range(topic_dist_rr.shape[0])), topic_dist_rr.index.tolist())
plt.tick_params(axis="y",labelsize=5)
plt.xlim(left=0)
plt.ylim(-.5, topic_dist_rr.shape[0]-.5)
plt.xlabel("Response Rate", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{topic_plot_dir}response_rate.png",dpi=200)
plt.close()

## Dependent Variables Over Time and Across Demographics
for dv in ["received_any_response","received_any_provider_response","received_physician_response","time_to_first_response"]:
    for variable, variable_map in zip(["gender_all","race_all","created_utc_weekday","created_utc_month"],[[gender_map],[race_map],[weekday_map],[month_map]]):
        ## Temporal
        fig, ax = plot_outcome_distribution(data_df,
                                            ivs=[variable],
                                            dv=dv,
                                            variable_maps=variable_map,
                                            time=True,
                                            iv_counts=False)
        fig.savefig(f"{demo_plot_dir}/temporal_{dv}_{variable}.png", dpi=200)
        plt.close(fig)
        ## Average
        fig, ax = plot_outcome_distribution(data_df,
                                            ivs=[variable],
                                            dv=dv,
                                            variable_maps=variable_map,
                                            time=False,
                                            iv_counts=False)
        fig.savefig(f"{demo_plot_dir}/{dv}_{variable}.png", dpi=200)
        plt.close(fig)

## Response as a Function of Length (measure of quality)
for textfield in ["selftext","title"]:
    for dv in ["received_any_response","received_any_provider_response","received_physician_response"]:
        fig, ax = plt.subplots(figsize=(10,5.8))
        for binr in [False,True]:
            binr_data = np.log(data_df.loc[data_df[dv]==binr][f"{textfield}_length"]+1)
            ax.hist(binr_data,
                    alpha=0.5,
                    label=binr,
                    density=True,
                    weights=np.ones(binr_data.shape[0])/len(binr_data),
                    bins=20)
        ax.set_xlabel(f"{textfield} Length", fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.set_title(dv.replace("_"," ").title(), fontsize=16, loc="left", fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/{textfield}_length_{dv}.png", dpi=200)
        plt.close(fig)

## Description Length as a function of Demographics and Topics
for textfield in ["selftext","title"]:
    for demo, demo_map in zip(["gender","race","age",],[[gender_map],[race_map],None]):
        fig, ax = plot_outcome_distribution(data_df,
                                            [f"{demo}_all"],
                                            f"{textfield}_length",
                                            variable_maps=demo_map,
                                            time=False)
        fig.savefig(f"{PLOT_DIR}/{textfield}_length_{demo}.png",dpi=200)
        plt.close(fig)
    ## Topic Distribution
    fig, ax = plot_outcome_distribution(data_df,
                                       ["max_topic"],
                                       f"{textfield}_length",
                                       variable_maps=[topic_map],
                                       time=False)
    ax.set_xlim(-.5, len(topic_map)-.5)
    ax.tick_params(axis="x", labelsize=4)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{textfield}_length_topics.png",dpi=200)
    plt.close(fig)
    
## Demographic Confusion Matrices (Post vs. Selftext)
for demo, demo_map in zip(["gender","race"],[gender_map,race_map]):
    demo_self = data_df[f"{demo}_selftext"].tolist()
    demo_title = data_df[f"{demo}_title"].tolist()
    if demo_map is not None:
        demo_self = list(map(lambda i: demo_map.get(i, "Unknown"), demo_self))
        demo_title = list(map(lambda i: demo_map.get(i, "Unknown"), demo_title))
    demo_all = [i for i in sorted(set(demo_self) | set(demo_title)) if i != "Unknown"] + ["Unknown"]
    demo_cm = metrics.confusion_matrix(demo_title, demo_self, labels=demo_all)
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.imshow(np.log(demo_cm + 1e-5), cmap=plt.cm.Blues, aspect="auto")
    for i, row in enumerate(demo_cm):
        for j, value in enumerate(row):
            ax.text(j, i, value if value > 0 else "", color="black", ha="center", va="center", rotation=45)
    ax.set_xticks(list(range(demo_cm.shape[0])))
    ax.set_xticklabels([d.replace("_"," ").title() for d in demo_all], rotation=45, ha="right")
    ax.set_yticks(list(range(demo_cm.shape[0])))
    ax.set_yticklabels([d.replace("_"," ").title() for d in demo_all])
    ax.set_title(f"{demo.title()} Placement", fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("Title", fontsize=14, fontweight="bold")
    ax.set_xlabel("Self-text", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{demo_plot_dir}{demo}_extraction_location.png",dpi=200)
    plt.close(fig)