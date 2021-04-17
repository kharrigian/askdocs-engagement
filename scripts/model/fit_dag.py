
"""

"""

## Plot Directory
PLOT_DIR = "./plots/model/dag/received_physician_response/all_covariates_aipw_missing/"

## Dependent Variable
# DEPENDENT_VARIABLE = "received_any_response"
DEPENDENT_VARIABLE = "received_physician_response"
# DEPENDENT_VARIABLE = "time_to_first_response"

## Dependent Variable Parameters
LOGTRANSFORM = True

## Independent Variable Parameters
DROP_NON_AUTOMOD = True
DROP_DEMO_LOC = True
DROP_DELETED = True
TOPIC_THRESH = 0.001
TOPIC_REPRESENTATION = "continuous" ## "continuous", "binary", "max"

###################
### Imports
###################

## Standard Libary
import os
import sys
from datetime import datetime

## External
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
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

def load_data():
    """

    """
    print("Loading Dataset")
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
    ## Log Transformation
    if LOGTRANSFORM and DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "continuous":
        Y[DEPENDENT_VARIABLE] = np.log(Y[DEPENDENT_VARIABLE])
    ## Topic Columns
    topic_columns = X.columns[[i for i, c in enumerate(X.columns) if c == "max_topic"][0]+1:].tolist()
    topic_columns_meets_thresh = ((X[topic_columns]!=0).mean(axis=0).sort_values() > TOPIC_THRESH)
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
    X = X[covariates].values
    Y = Y[DEPENDENT_VARIABLE].values
    return X, Y, covariates, topic_columns


def construct_directed_graph(covariates,
                             topic_variables):
    """

    Demos (Age, Gender, Race) -> Topic, Temporal, Description Quality, Response
    Participation -> Description, Response
    Topic -> Description Quality, Response
    Description Quality -> Response
    Temporal -> Response
    """
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
    ## Generate the DAG
    p = len(labels)
    D = np.zeros((p, p),dtype=int)
    for c1, c1name in enumerate(labels):
        c1_type = label_types[c1]
        for c2_type in DAG_map[c1_type]:
            c2_values = label_types_map[c2_type]
            for c2v in c2_values:
                D[c1, labels2ind[c2v]] = 1
    return D

def directed_to_edges(D, vertices):
    """

    """
    edges = []
    for r, row in enumerate(D):
        row_nn = row.nonzero()[0]
        for nn in row_nn:
            edges.append((vertices[r], vertices[nn]))
    return edges

def convert_to_one_hot(X,
                       covariate,
                       missingness_col=None,
                       keep_all=False):
    """

    """
    ## Identify Unique Targets
    targets = sorted(X[covariate].unique())
    targets = [int(t) for t in targets if not np.isnan(t)]
    ## Generally, only need k-1 Dimensions to represent K classes
    if not keep_all:
        targets = targets[1:]
    ## Construct Encodings
    X_one_hot = [(X[covariate]==target).astype(int).to_frame(f"{covariate}_{target}") for target in targets]
    X_one_hot = pd.concat(X_one_hot, axis=1)
    one_hot_columns = X_one_hot.columns.tolist()
    ## Update Encodings Based on Missing Data if Appropriate
    if missingness_col is not None:
        X_one_hot.loc[(X[missingness_col] == 0)] *= np.nan
        X_missing = [X[missingness_col].to_frame(f"R_{col}") for col in X_one_hot.columns]
        X_missing = pd.concat(X_missing, axis=1)
        X = pd.merge(X, X_missing, left_index=True, right_index=True)
        X = X.drop(missingness_col, axis=1)
    ## Merge Encodings with Original Dataset
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

def estimate_glm_influence(fit_model,
                           x,
                           dv_type,
                           vary_cols,
                           vary_cols_labels=None,
                           n_samples=1000,
                           random_state=42):
    """
    Estimate influence of certain covariates on model output by comparing
    mean prediction with predictions generated by systematically turning-on
    features

    Reference:
        Making the Most of Statistical Analyses: Improving Interpretation and Presentation
        Author(s): Gary King, Michael Tomz and Jason Wittenberg

    Args:
        fit_model (GLM): Trained linear model
        x (pandas DataFrame): Input data
        dv_type (str): Whether the dependent variable is binary or continuous
        vary_cols (list): Which variables should be varied
        vary_cols_labels (list): If desired, alternative names for the vary_cols
        n_samples (int): Number of samples from multivariate normal distribution to generate
        random_state (int): Random seed
    """
    ## Set Random State
    if random_state:
        np.random.seed(random_state)
    ## Extract Model Parameters and Sample from Multivariate Normal
    model_covariance = fit_model.cov_params()
    model_means = fit_model.params
    W = stats.multivariate_normal(model_means,
                                  model_covariance).rvs(n_samples)
    ## Format Labels
    if vary_cols_labels is None:
        vary_cols_labels = vary_cols
    ## Isolate Means and Set Vary Columns to Reference Value
    x = x.copy()
    x_mean = x.mean(axis=0)
    ## Compute Baseline Probabilities
    baseline_input_array = np.hstack([np.r_[1], x_mean.values]).reshape(1,-1)
    baseline_yhat = np.matmul(W, baseline_input_array.T).T[0]
    if dv_type == "binary":
        baseline_yhat = 1 / (1 + np.exp(-baseline_yhat))
    baseline_q_hat = np.percentile(baseline_yhat, q=[2.5,50,97.5])
    confidence_intervals = [baseline_q_hat]
    ## Compute Implict Class Probability Probability
    x_mean[vary_cols] = 0
    implicit_input_array = np.hstack([np.r_[1], x_mean.values]).reshape(1,-1)
    implicit_yhat = np.matmul(W, implicit_input_array.T).T[0]
    if dv_type == "binary":
        implicit_yhat = 1 / (1 + np.exp(-implicit_yhat))
    implicit_q_hat = np.percentile(implicit_yhat, q=[2.5,50,97.5])
    confidence_intervals.append(implicit_q_hat)
    ## Columpute Varied Probabilities
    for col in vary_cols:
        col_x_mean = x_mean.copy()
        col_x_mean[col] = 1
        input_array = np.hstack([np.r_[1], col_x_mean.values]).reshape(1,-1)
        col_yhat = np.matmul(W, input_array.T).T[0]
        if dv_type == "binary":
            col_yhat = 1 / (1 + np.exp(-col_yhat))
        q_hat = np.percentile(col_yhat, q=[2.5,50,97.5])
        confidence_intervals.append(q_hat)
    ## Format Confidence Intervals
    confidence_intervals = pd.DataFrame(confidence_intervals,
                                        index=["baseline","unknown"] + vary_cols_labels,
                                        columns=["lower","median","upper"])
    return confidence_intervals

def shaded_bar_plot(dataframe,
                    median_col,
                    lower_col="lower",
                    upper_col="upper",
                    xlabel=None,
                    title=None):
    """

    """
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.hlines(list(range(dataframe.shape[0])),
              xmin=dataframe[lower_col].min()-0.01,
              xmax=dataframe[upper_col].max()+0.01,
              linestyle=":",
              alpha=0.4, 
              zorder=-1,
              linewidth=0.5)
    ax.axvline(0, color="black", linestyle="-", alpha=0.5)
    ax.barh(list(range(dataframe.shape[0])),
            left=dataframe[lower_col],
            width=dataframe[upper_col]-dataframe[lower_col],
            color="C0",
            alpha=0.2)
    ax.scatter(dataframe[median_col],
               list(range(dataframe.shape[0])), color="navy")
    ax.set_yticks(list(range(dataframe.shape[0])))
    kwargs = {"fontsize":5} if dataframe.shape[0] > 30 else {}
    ax.set_yticklabels([i.replace("_"," ").title() for i in dataframe.index.tolist()], **kwargs)
    ax.set_xlim(dataframe[lower_col].min()-0.01, dataframe[upper_col].max()+0.01)
    ax.set_ylim(-.5, dataframe.shape[0]-.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold", fontstyle="italic")
    fig.tight_layout()
    return fig, ax

def estimate_causal_effects(dag_model,
                            data,
                            variable,
                            categories,
                            n_bootstrap=10,
                            alpha=0.05):
    """
    Estimate causal effect (ACE or Odds Ratio) using Ananke package implementation

    Args:
        dag_model (graph): Ananke graph
        variable (str): Treatment variable (e.g. race_all)
        categories (list): One-hot encoding categories appended to variable string
        n_bootstrap (int): Number of bootstraps to use
        alpha (float): Confidence level
    """
    ce_estimates = []
    for c, category in enumerate(categories):
        if f"{variable}_{category}" not in dag_model.vertices:
            continue
        print(f"Computing Effect for Group {c+1}/{len(categories)}: {category}")
        dag_ce = estimation.CausalEffect(dag_model, f"{variable}_{category}", DEPENDENT_VARIABLE)
        dag_ce_est = dag_ce.compute_effect(data, "aipw", n_bootstraps=n_bootstrap, alpha=alpha)
        dag_ce_or = np.exp(np.array(dag_ce_est))
        ce_estimates.append([category] + list(dag_ce_or))
    ce_estimates = pd.DataFrame(ce_estimates,
                                columns=["category","median","lower","upper"]).set_index("category")
    return ce_estimates

def _aipw(data_obs,
          frequency_weights,
          dag_ce,
          assignment,
          binary=True):
    """
    AIPW with Weights
    """
    ## Format Indices
    data_obs_r = data_obs.reset_index(drop=True)
    frequency_weights_r = frequency_weights.reset_index(drop=True)
    ## Extract Markov Pillow
    Y = data_obs_r[dag_ce.outcome]
    mp_T = dag_ce.graph.markov_pillow([dag_ce.treatment], dag_ce.p_order)
    ## Fit Treatment Model
    if len(mp_T) != 0:
        formula_T = dag_ce.treatment + " ~ " + "+".join(mp_T)
        model = sm.GLM.from_formula(formula_T,
                                    data=data_obs_r,
                                    freq_weights=frequency_weights_r,
                                    family=sm.families.Binomial()).fit()
        prob_T = model.predict(data_obs_r)
        formula_Y = dag_ce.outcome + " ~ " + dag_ce.treatment + '+' + '+'.join(mp_T)
    else:
        prob_T = np.ones(len(data_obs_r)) * np.mean(data_obs_r[dag_ce.treatment])
        formula_Y = dag_ce.outcome + " ~ " + dag_ce.treatment
    ## Indices
    indices_T0 = data_obs_r.index[data_obs_r[dag_ce.treatment] == 0]
    prob_T[indices_T0] = 1 - prob_T[indices_T0]
    indices = data_obs_r[dag_ce.treatment] == assignment
    ## Assignment
    data_assign = data_obs_r.copy()
    data_assign[dag_ce.treatment] = assignment
    ## Outcome Model
    if binary:
        model = sm.GLM.from_formula(formula_Y,
                                    data=data_obs_r,
                                    freq_weights=frequency_weights_r,
                                    family=sm.families.Binomial()).fit()
    else:
        model = sm.GLM.from_formula(formula_Y,
                                    data=data_obs_r,
                                    freq_weights=frequency_weights_r,
                                    family=sm.families.Gaussian()).fit()
    ## Estimate
    Yhat_vec = model.predict(data_assign)
    point_est = np.mean((indices / prob_T) * (Y - Yhat_vec) + Yhat_vec)
    return point_est

def compute_effect(graph_,
                   treatment,
                   outcome,
                   data_obs,
                   freq_weights,
                   binary,
                   n_bootstrap=0,
                   alpha=0.05,
                   random_state=42):
    """
    Compute Effects using AIPW
    """
    ## Initialize Graph
    dag_ce = estimation.CausalEffect(graph_, treatment, outcome)
    ## Compute General ACE
    point_estimate_T1 = _aipw(data_obs, freq_weights, dag_ce, 1, binary=binary)
    point_estimate_T0 = _aipw(data_obs, freq_weights, dag_ce, 0, binary=binary)
    if binary:
        ace = np.log((point_estimate_T1/(1-point_estimate_T1))/(point_estimate_T0/(1-point_estimate_T0)))
    else:
        ace = point_estimate_T1 - point_estimate_T0
    if n_bootstrap == 0:
        return ace
    ## Bootstrap
    ace_vec = []
    if random_state:
        np.random.seed(random_state)
    for sample in range(n_bootstrap):
        ## Resample Data and Weights
        data_obs_sample = data_obs.sample(len(data_obs), replace=True)
        freq_weights_sample = freq_weights.loc[data_obs_sample.index].copy()
        ## Estimate ACE in resampled data
        estimate_T1 = _aipw(data_obs_sample, freq_weights_sample, dag_ce, 1, binary=binary)
        estimate_T0 = _aipw(data_obs_sample, freq_weights_sample, dag_ce, 0, binary=binary)
        if binary:
            ace_vec.append(np.log((estimate_T1/(1-estimate_T1))/(estimate_T0/(1-estimate_T0))))
        else:
            ace_vec.append(estimate_T1 - estimate_T0)
    q_low = np.nanpercentile(ace_vec, 100 * alpha / 2)
    q_high = np.nanpercentile(ace_vec, 100 - 100 * alpha / 2)
    return ace, q_low, q_high

def estimate_causal_effects_missing(dag_model,
                                    data_obs,
                                    freq_weights,
                                    variable,
                                    categories,
                                    n_bootstrap=10,
                                    alpha=0.05):
    """
    Estimate causal effect (ACE or Odds Ratio) using Ananke package implementation

    Args:
        dag_model (graph): Ananke graph
        variable (str): Treatment variable (e.g. race_all)
        categories (list): One-hot encoding categories appended to variable string
        est_method (str): How to estimate the causal effect
        n_bootstrap (int): Number of bootstraps to use
        alpha (float): Confidence level
    """
    ce_estimates = []
    for c, category in enumerate(categories):
        if f"{variable}_{category}" not in dag_model.vertices:
            continue
        print(f"Computing Effect for Group {c+1}/{len(categories)}: {category}")
        dag_ce_est = compute_effect(graph_=dag_model,
                                    treatment=f"{variable}_{category}",
                                    outcome=DEPENDENT_VARIABLE,
                                    data_obs=data_obs,
                                    freq_weights=freq_weights,
                                    binary=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE]=="binary",
                                    n_bootstrap=n_bootstrap,
                                    alpha=alpha,
                                    random_state=42)
        dag_ce_or = np.exp(np.array(dag_ce_est))
        ce_estimates.append([category] + list(dag_ce_or))
    ce_estimates = pd.DataFrame(ce_estimates,
                                columns=["category","median","lower","upper"]).set_index("category")
    return ce_estimates

def get_data_df(X,
                Y,
                covariates,
                topic_columns,
                demos_only=False,
                keep_all_one_hot=False):
    """

    """
    ## Data Formatting
    data = pd.DataFrame(np.hstack([X, Y.reshape(-1,1)]),
                        columns=covariates+[DEPENDENT_VARIABLE])
    if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
        data[DEPENDENT_VARIABLE] = data[DEPENDENT_VARIABLE].astype(int)
    ## Isolate Demographics
    if demos_only:
        demo_cols = [c for c in covariates if classify_covariate(c, topic_columns).startswith("demo_")]
        data = data[demo_cols + [DEPENDENT_VARIABLE]]
    ## One Hot Encoding
    for col in list(data.columns):
        if col == DEPENDENT_VARIABLE:
            continue
        if IV_VARIABLE_TYPES[col] in ["categorical","binary"]:
            data[col] = data[col].astype(int)
            data, _ = convert_to_one_hot(data,
                                         col,
                                         missingness_col=None,
                                         keep_all=keep_all_one_hot)
    ## Data Prep
    y = data[DEPENDENT_VARIABLE].values
    x = data.drop(DEPENDENT_VARIABLE,axis=1).values
    x = sm.add_constant(x)
    features = data.drop(DEPENDENT_VARIABLE,axis=1).columns.tolist()
    return data, features, x, y

def get_data_df_missing(X,
                        Y,
                        covariates,
                        topic_columns,
                        keep_missing=False,
                        keep_all_one_hot=False):
    """

    """
    ## Data Formatting
    data = pd.DataFrame(np.hstack([X, Y.reshape(-1,1)]),
                        columns=covariates+[DEPENDENT_VARIABLE])
    if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
        data[DEPENDENT_VARIABLE] = data[DEPENDENT_VARIABLE].astype(int)
    ## Append Missingness Indicators
    if not keep_missing:
        missingness_cols = {}
        for col in (data == -1).any(axis=0).loc[(data == -1).any(axis=0)].index.tolist():
            data[f"R_{col}"] = (data[f"{col}"] != -1).astype(int)
            data[f"{col}"] = data[f"{col}"].replace(-1,np.nan)
            missingness_cols[f"{col}"] = f"R_{col}"
        ## One Hot Encoding
        missing_value_indicators = list(missingness_cols.values())
        for col in list(data.columns):
            if col == DEPENDENT_VARIABLE or col in missing_value_indicators:
                continue
            if IV_VARIABLE_TYPES[col] in ["categorical","binary"]:
                data, oh_columns = convert_to_one_hot(data,
                                                      col,
                                                      missingness_col=missingness_cols.get(col,None),
                                                      keep_all=keep_all_one_hot)
                if missingness_cols.get(col,None) is None:
                    continue
                _ = missingness_cols.pop(col,None)
                for oh_col in oh_columns:
                    missingness_cols[oh_col] = f"R_{oh_col}"
    else:
        for col in list(data.columns):
            if col == DEPENDENT_VARIABLE:
                continue
            if IV_VARIABLE_TYPES[col] in ["categorical","binary"]:
                data[col] = data[col].astype(int)
                data, _ = convert_to_one_hot(data,
                                             col,
                                             missingness_col=None,
                                             keep_all=keep_all_one_hot)
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
    ## Early Return
    if keep_missing:
        features = [d for d in data.columns if not d.startswith("R_") and d != DEPENDENT_VARIABLE]
        return data, None, None, features, topic_cols_clean
    ## Isolate Observed Data
    data_obs = data.dropna()
    ## Adjustment Edges (Missingness Caused By These Factor Types)
    parents_map = {
        "demo_gender":["demo_race","demo_age","topic"],
        "demo_race":["demo_gender","demo_age","topic"],
        "demo_age":["demo_gender","demo_race","topic"]
    }
    ## Get Columns and Their Types
    nonmissingness_cols = [d for d in data.columns if not d.startswith("R_") and d != DEPENDENT_VARIABLE]
    col_type_map = {}
    for nmc in nonmissingness_cols:
        nmc_type = classify_covariate(nmc, topic_cols_clean)
        if nmc_type not in col_type_map:
            col_type_map[nmc_type] = []
        col_type_map[nmc_type].append(nmc)
    ## Compute Missingness Probabilities
    missingness_probabilities = []
    for missing_col in tqdm(missingness_cols.keys(), desc="Computing Missingness Probabilites", total=len(missingness_cols)):
        missing_col_out = missingness_cols[missing_col]
        missing_col_type = classify_covariate(missing_col, topic_cols_clean)
        parent_vars = [col_type_map[j] for j in parents_map[missing_col_type]]
        parent_vars = [l for i in parent_vars for l in i]
        data_obs_missing_col = data.dropna(subset=parent_vars)
        model_missing_col = sm.GLM.from_formula(formula="{} ~ {}".format(missing_col_out, "+".join(parent_vars)),
                                                data=data_obs_missing_col,
                                                family=sm.families.Binomial()).fit()
        proba_missing_col = model_missing_col.predict(data_obs)
        missingness_probabilities.append(proba_missing_col)
    ## Compute Adjustment Weights
    frequency_weights = 1 / np.product(pd.concat(missingness_probabilities, axis=1).values, axis=1)
    frequency_weights = pd.Series(frequency_weights, index=data_obs.index)
    return data, data_obs, frequency_weights, nonmissingness_cols, topic_cols_clean

def fit_glm(x,
            y,
            feature_names):
    """

    """
    ## Get Dependent Variable Family
    if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] in ["continuous","ordinal"]:
        family = sm.families.Gaussian()
    elif DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
        family = sm.families.Binomial()
    else:
        raise ValueError("Variable type not recognized")
    model = sm.GLM(endog=y, exog=x, family=family).fit(maxiter=1000, full_output=True)
    coefs = pd.DataFrame(np.hstack([model.params.reshape(-1,1), model.conf_int()]), 
                         index=["intercept"]+feature_names,
                         columns=["coef","lower","upper"]).sort_values("coef").drop("intercept")
    return model, coefs

def draw_simple(covariates,
                topic_columns):
    """

    """
    print("Visualizing Simplified Graphical Model")
    ## Create Directed Graph (Visualization - Collapse Encodings)
    covariates_vis = sorted(set(list(map(lambda c: c if c not in topic_columns else "topic", covariates))))
    D_vis = construct_directed_graph(covariates_vis, ["topic"])
    D_edges_vis = directed_to_edges(D_vis, covariates_vis+[DEPENDENT_VARIABLE])
    ## Create Graph
    dag_vis = graphs.DAG(vertices=covariates_vis+[DEPENDENT_VARIABLE], di_edges=D_edges_vis)
    dag_vis.draw().render(f"{PLOT_DIR}/dag_simple.gv", view=False)


def demographics_only_model(X, Y, covariates, topic_columns):
    """

    """
    print("Modeling Data: Non-graphical Approach (Demographics Only)")
    ## Construct Data
    data, demo_features, x, y = get_data_df(X,
                                            Y,
                                            covariates,
                                            topic_columns,
                                            demos_only=True)
    ## Model Fitting
    demo_model, demo_coefs = fit_glm(x, y, demo_features)
    ## Cache Coefficients
    demo_coefs.to_csv(f"{PLOT_DIR}demographics_only_glm_coefficients.csv")
    ## Feature Groups
    feature_groups = {"age":[],
                      "gender":[],
                      "race":[]}
    for feature in demo_features:
        for group in feature_groups:
            if feature.startswith(group):
                feature_groups[group].append(feature)
                break
    ## Plot Coefficients
    for variable_type, variable_type_features in feature_groups.items():
        plot_coefs = demo_coefs.loc[variable_type_features].copy()
        plot_coefs.index =  format_feature_names(variable_type_features, variable_type)
        fig, ax = shaded_bar_plot(plot_coefs,
                                "coef",
                                xlabel="Coefficient",
                                title=variable_type.replace("_"," ").title())
        fig.savefig(f"{PLOT_DIR}demographics_only_glm_coefficient_{variable_type}.png", dpi=200)
        plt.close(fig)
    ## Estimate Race and Gender Influence
    race_ci = estimate_glm_influence(fit_model=demo_model,
                                     x=data[demo_features],
                                     dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                     vary_cols=feature_groups["race"],
                                     vary_cols_labels=format_feature_names(feature_groups["race"], "race"),
                                     n_samples=100,
                                     random_state=42)
    gender_ci = estimate_glm_influence(fit_model=demo_model,
                                       x=data[demo_features],
                                       dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                       vary_cols=feature_groups["gender"],
                                       vary_cols_labels=format_feature_names(feature_groups["gender"], "gender"),
                                       n_samples=100,
                                       random_state=42)
    ## Cache Influence Effects
    gender_ci.to_csv(f"{PLOT_DIR}demographics_only_glm_demographic_influence_gender.csv")
    race_ci.to_csv(f"{PLOT_DIR}demographics_only_glm_demographic_influence_race.csv")
    ## Plot Influence Effects
    for ci_df, ci_name in zip([race_ci,gender_ci],["race","gender"]):
        fig, ax = shaded_bar_plot(ci_df,
                                  median_col="median",
                                  xlabel="Predicted Probability" if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary" else \
                                         "Predicted {}".format(DEPENDENT_VARIABLE.replace("_"," ").title()),
                                  title=f"Demographic Type: {ci_name.title()}")
        ax.axvline(ci_df.loc["baseline","median"], alpha=0.2, linestyle=":", color="black")
        fig.savefig(f"{PLOT_DIR}demographics_only_glm_demographic_influence_{ci_name}.png", dpi=200)
        plt.close(fig)

def all_covariates_model(X, Y, covariates, topic_columns):
    """

    """
    print("Modeling Data: Non-Graphical Approach (All Covariates)")
    ## Data Formatting
    data, features, x, y = get_data_df(X,
                                       Y,
                                       covariates,
                                       topic_columns,
                                       demos_only=False)
    ## Fit Model
    all_model, all_coefs = fit_glm(x, y, features)
    ## Cache Coefficients
    all_coefs.to_csv(f"{PLOT_DIR}all_covariates_glm_coefficients.csv")
    ## Feature Groups
    feature_groups = {"age":[],
                      "gender":[],
                      "race":[],
                      "created_utc_hour":[],
                      "created_utc_weekday":[],
                      "created_utc_month":[],
                      "created_utc_year":[],
                      "activity":[],
                      "topic":[]}
    for feature in features:
        for group in feature_groups:
            if feature.startswith(group):
                feature_groups[group].append(feature)
                break
            elif feature in topic_columns or feature.startswith("max_topic"):
                feature_groups["topic"].append(feature)
                break
            elif feature in ["n_prior_submissions","is_automod_format"] or feature.endswith("_length"):
                feature_groups["activity"].append(feature)
                break
    ## Plot Coefficients
    for variable_type, variable_type_features in feature_groups.items():
        feature_labels = format_feature_names(variable_type_features, variable_type)
        plot_coefs = all_coefs.loc[variable_type_features].copy()
        plot_coefs.index = feature_labels
        fig, ax = shaded_bar_plot(plot_coefs,
                                  "coef",
                                  xlabel="Coefficient",
                                  title=variable_type.replace("_"," ").title())
        fig.savefig(f"{PLOT_DIR}all_covariates_glm_coefficient_{variable_type}.png", dpi=200)
        plt.close(fig)
    ## Estimate Race and Gender Influence
    race_ci = estimate_glm_influence(fit_model=all_model,
                                     x=data[features],
                                     dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                     vary_cols=feature_groups["race"],
                                     vary_cols_labels=format_feature_names(feature_groups["race"], "race"),
                                     n_samples=100,
                                     random_state=42)
    gender_ci = estimate_glm_influence(fit_model=all_model,
                                       x=data[features],
                                       dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                       vary_cols=feature_groups["gender"],
                                       vary_cols_labels=format_feature_names(feature_groups["gender"], "gender"),
                                       n_samples=100,
                                       random_state=42)
    ## Cache Influence Effects
    gender_ci.to_csv(f"{PLOT_DIR}all_covariates_glm_demographic_influence_gender.csv")
    race_ci.to_csv(f"{PLOT_DIR}all_covariates_glm_demographic_influence_race.csv")
    ## Plot Influence Effects
    for ci_df, ci_name in zip([race_ci,gender_ci],["race","gender"]):
        fig, ax = shaded_bar_plot(ci_df,
                                  median_col="median",
                                  xlabel="Predicted Probability" if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary" else "Predicted {}".format(DEPENDENT_VARIABLE.replace("_"," ").title()),
                                  title=f"Demographic Type: {ci_name.title()}")
        ax.axvline(ci_df.loc["baseline","median"], alpha=0.2, linestyle=":", color="black")
        fig.savefig(f"{PLOT_DIR}all_covariates_glm_demographic_influence_{ci_name}.png", dpi=200)
        plt.close(fig)

def all_covariates_missing_model(X, Y, covariates, topic_columns):
    """

    """
    print("Modeling Data: Non-Graphical Approach (All Covariates + Missingness)")
    ## Get Data
    data, data_obs, frequency_weights, nonmissingness_cols, topic_cols_clean = get_data_df_missing(X,
                                                                                                   Y,
                                                                                                   covariates,
                                                                                                   topic_columns,
                                                                                                   keep_missing=False)
    ## Fit Model
    adjusted_model = sm.GLM.from_formula(formula="{} ~ {}".format(DEPENDENT_VARIABLE, "+".join(nonmissingness_cols)),
                                         data=data_obs,
                                         freq_weights=frequency_weights,
                                         family=sm.families.Binomial() if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] != "continuous" else \
                                                sm.families.Gaussian())
    adjusted_model_fit = adjusted_model.fit(maxiter=1000, full_output=True)
    adjusted_coefs = pd.DataFrame(np.hstack([adjusted_model_fit.params.values.reshape(-1,1), adjusted_model_fit.conf_int()]), 
                                  index=["intercept"]+nonmissingness_cols,
                                  columns=["coef","lower","upper"]).sort_values("coef").drop("intercept")
    ## Cache Coefficients
    adjusted_coefs.to_csv(f"{PLOT_DIR}adjusted_glm_coefficients.csv")
    ## Feature Groups
    feature_groups = {"age":[],
                      "gender":[],
                      "race":[],
                      "created_utc_hour":[],
                      "created_utc_weekday":[],
                      "created_utc_month":[],
                      "created_utc_year":[],
                      "activity":[],
                      "topic":[]}
    for feature in nonmissingness_cols:
        for group in feature_groups:
            if feature.startswith(group):
                feature_groups[group].append(feature)
                break
            elif feature in topic_cols_clean or feature.startswith("max_topic"):
                feature_groups["topic"].append(feature)
                break
            elif feature in ["n_prior_submissions","is_automod_format"] or feature.endswith("_length"):
                feature_groups["activity"].append(feature)
                break
    ## Plot Coefficients
    for variable_type, variable_type_features in feature_groups.items():
        plot_coefs = adjusted_coefs.loc[variable_type_features].copy()
        fig, ax = shaded_bar_plot(plot_coefs,
                                  "coef",
                                  xlabel="Coefficient",
                                  title=variable_type.replace("_"," ").title())
        fig.savefig(f"{PLOT_DIR}adjusted_glm_coefficient_{variable_type}.png", dpi=200)
        plt.close(fig)
    ## Estimate Influence in Adjusted Model
    adj_race_ci = estimate_glm_influence(fit_model=adjusted_model_fit,
                                         x=data_obs[nonmissingness_cols],
                                         dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                         vary_cols=feature_groups["race"],
                                         n_samples=100,
                                         random_state=42)
    adj_gender_ci = estimate_glm_influence(fit_model=adjusted_model_fit,
                                           x=data_obs[nonmissingness_cols],
                                           dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                           vary_cols=feature_groups["gender"],
                                           n_samples=100,
                                           random_state=42)
    adj_race_ci.index = adj_race_ci.index.map(lambda i: i if i != "unknown" else race_map[0])
    adj_gender_ci.index = adj_gender_ci.index.map(lambda i: i if i != "unknown" else gender_map[0])
    ## Cache Influence
    adj_gender_ci.to_csv(f"{PLOT_DIR}adjusted_glm_demographic_influence_gender.csv")
    adj_race_ci.to_csv(f"{PLOT_DIR}adjusted_glm_demographic_influence_race.csv")
    ## Plot Influence Effects
    for ci_df, ci_name in zip([adj_race_ci,adj_gender_ci],["race","gender"]):
        fig, ax = shaded_bar_plot(ci_df,
                                median_col="median",
                                xlabel="Predicted Probability" if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary" else "Predicted {}".format(DEPENDENT_VARIABLE.replace("_"," ").title()),
                                title=f"Demographic Type: {ci_name.title()}")
        ax.axvline(ci_df.loc["baseline","median"], alpha=0.2, linestyle=":", color="black")
        fig.savefig(f"{PLOT_DIR}adjusted_glm_demographic_influence_{ci_name}.png", dpi=200)
        plt.close(fig)

def all_covariates_aipw_model(X, Y, covariates, topic_columns):
    """

    """
    print("Modeling Data: Graphical Approach (AIPW w/o Missingness)")
    ## Get Data
    data, _, _, features, topic_cols_clean = get_data_df_missing(X,
                                                                 Y,
                                                                 covariates,
                                                                 topic_columns,
                                                                 keep_missing=True,
                                                                 keep_all_one_hot=True)
    ## Create Directed Graph
    D = construct_directed_graph(features, set(topic_cols_clean))
    D_edges = directed_to_edges(D, features+[DEPENDENT_VARIABLE])
    ## Initial Graphical Model
    dag_model = graphs.DAG(features+[DEPENDENT_VARIABLE], D_edges)
    ## Causal Effect Estimation
    gender_ce = estimate_causal_effects(dag_model,
                                        data,
                                        "gender_all",
                                        ["unknown","male","female"],
                                        n_bootstrap=100,
                                        alpha=0.05)
    race_ce = estimate_causal_effects(dag_model,
                                      data,
                                      "race_all",
                                      ["unknown","white","black","hispanic","asian","indian","mixed","middle_eastern","pacific_islander"],
                                      n_bootstrap=100,
                                      alpha=0.05)
    ## Cache Causal Effects
    gender_ce.to_csv(f"{PLOT_DIR}dag_demographic_causal_effect_gender.csv")
    race_ce.to_csv(f"{PLOT_DIR}dag_demographic_causal_effect_race.csv")
    ## Plot Causal Effects
    for ci_df, ci_name in zip([race_ce,gender_ce],["race","gender"]):
        fig, ax = shaded_bar_plot(ci_df,
                                median_col="median",
                                xlabel="Odds Ratio" if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary" else "Average Causal Effect (ACE)",
                                title=f"Demographic Type: {ci_name.title()}")
        if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
            ax.axvline(1, alpha=0.2, color="black", zorder=-1)
        fig.savefig(f"{PLOT_DIR}dag_demographic_causal_effect_{ci_name}.png", dpi=200)
        plt.close(fig)

def all_covariates_aipw_missing_model(X, Y, covariates, topic_columns):
    """

    """
    print("Modeling Data: Graphical Approach (AIPW w/ Missingness)")
    ## Get Data
    data, data_obs, frequency_weights, nonmissingness_cols, topic_cols_clean = get_data_df_missing(X,
                                                                                                   Y,
                                                                                                   covariates,
                                                                                                   topic_columns,
                                                                                                   keep_missing=False,
                                                                                                   keep_all_one_hot=True)
    ## Create Directed Graph
    D_nonmissing = construct_directed_graph(nonmissingness_cols, set(topic_cols_clean))
    D_edges_nonmissing = directed_to_edges(D_nonmissing, nonmissingness_cols+[DEPENDENT_VARIABLE])
    ## Initialize Graphical Model
    dag_model_nonmissing = graphs.DAG(nonmissingness_cols+[DEPENDENT_VARIABLE],
                                      D_edges_nonmissing)
    ## Causal Effect Estimation
    gender_ce = estimate_causal_effects_missing(dag_model_nonmissing,
                                                data_obs=data_obs,
                                                freq_weights=frequency_weights,
                                                variable="gender_all",
                                                categories=["male","female"],
                                                n_bootstrap=100,
                                                alpha=0.05)
    race_ce = estimate_causal_effects_missing(dag_model_nonmissing,
                                              data_obs=data_obs,
                                              freq_weights=frequency_weights,
                                              variable="race_all",
                                              categories=["white","black","hispanic","asian","indian","mixed","middle_eastern","pacific_islander"],
                                              n_bootstrap=100,
                                              alpha=0.05)
    ## Cache Causal Effects
    gender_ce.to_csv(f"{PLOT_DIR}dag_aipw_demographic_causal_effect_gender.csv")
    race_ce.to_csv(f"{PLOT_DIR}dag_aipw_demographic_causal_effect_race.csv")
    ## Plot Causal Effects
    for ci_df, ci_name in zip([race_ce,gender_ce],["race","gender"]):
        fig, ax = shaded_bar_plot(ci_df,
                                median_col="median",
                                xlabel="Odds Ratio" if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary" else "Average Causal Effect (ACE)",
                                title=f"Demographic Type: {ci_name.title()}")
        if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
            ax.axvline(1, alpha=0.2, color="black", zorder=-1)
        fig.savefig(f"{PLOT_DIR}dag_aipw_demographic_causal_effect_{ci_name}.png", dpi=200)
        plt.close(fig)

def main():
    """

    """
    ## Load Data
    X, Y, covariates, topic_columns = load_data()
    ## Visualize DAG
    if sys.argv[1] in ["all_covariates_aipw","all_covariates_missing_aipw"]:
        _ = draw_simple(covariates, topic_columns)
    ## Modeling
    if sys.argv[1] == "demographics_only":
        _ = demographics_only_model(X, Y, covariates, topic_columns)
    elif sys.argv[1] == "all_covariates":
        _ = all_covariates_model(X, Y, covariates, topic_columns)
    elif sys.argv[1] == "all_covariates_aipw":
        _ = all_covariates_aipw_model(X, Y, covariates, topic_columns)
    elif sys.argv[1] == "all_covariates_missing":
        _ = all_covariates_missing_model(X, Y, covariates, topic_columns)
    elif sys.argv[1] == "all_covariates_missing_aipw":
        _ = all_covariates_aipw_missing_model(X, Y, covariates, topic_columns)
    else:
        raise ValueError("Did not understand model request from command line.")

####################
### Execute
####################

if __name__ == "__main__":
    _ = main()