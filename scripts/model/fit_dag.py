
"""

"""

## Plot Directory
PLOT_DIR = "./plots/model/dag/received_any_response/"

## Dependent Variable
DEPENDENT_VARIABLE = "received_any_response"
# DEPENDENT_VARIABLE = "received_physician_response"
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

def estimate_glm_influence(fit_model,
                           x,
                           dv_type,
                           vary_cols,
                           vary_cols_labels=None,
                           n_samples=1000,
                           random_state=42):
    """
    Reference:
        Making the Most of Statistical Analyses: Improving Interpretation and Presentation
        Author(s): Gary King, Michael Tomz and Jason Wittenberg
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
    x_mean[vary_cols] = 0
    ## Compute Baseline Probabilities
    baseline_input_array = np.hstack([np.r_[1], x_mean.values]).reshape(1,-1)
    baseline_yhat = np.matmul(W, baseline_input_array.T).T[0]
    if dv_type == "binary":
        baseline_yhat = 1 / (1 + np.exp(-baseline_yhat))
    baseline_q_hat = np.percentile(baseline_yhat, q=[2.5,50,97.5])
    confidence_intervals = [baseline_q_hat]
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
                                        index=["baseline"] + vary_cols_labels,
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
                            variable,
                            categories,
                            est_method="aipw",
                            n_bootstrap=10,
                            alpha=0.05):
    ce_estimates = []
    for c, category in enumerate(categories):
        if f"{variable}_{category}" not in dag_model.vertices:
            continue
        print(f"Computing Effect for Group {c+1}/{len(categories)}: {category}")
        dag_ce = estimation.CausalEffect(dag_model, f"{variable}_{category}", DEPENDENT_VARIABLE)
        dag_ce_est = dag_ce.compute_effect(data, est_method, n_bootstraps=n_bootstrap, alpha=alpha)
        dag_ce_or = np.exp(np.array(dag_ce_est))
        ce_estimates.append([category] + list(dag_ce_or))
    ce_estimates = pd.DataFrame(ce_estimates,
                                columns=["category","median","lower","upper"]).set_index("category")
    return ce_estimates

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

########################
### Non-graphical Model (Generalized Linear Model)
########################

print("Modeling Data: Non-Graphical Approach")

## Data Formatting
data = pd.DataFrame(np.hstack([X, Y.reshape(-1,1)]),
                    columns=covariates+[DEPENDENT_VARIABLE])
if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
    data[DEPENDENT_VARIABLE] = data[DEPENDENT_VARIABLE].astype(int)

## One Hot Encoding
for col in list(data.columns):
    if col == DEPENDENT_VARIABLE:
        continue
    if IV_VARIABLE_TYPES[col] in ["categorical","binary"]:
        data[col] = data[col].astype(int)
        data, _ = convert_to_one_hot(data, col)

## Data Scaling
for covariate in data.columns.tolist():
    if covariate == DEPENDENT_VARIABLE or covariate[-1].isdigit():
        continue
    data[covariate] = (data[covariate] - data[covariate].mean()) / data[covariate].std()

## Data Prep
y = data[DEPENDENT_VARIABLE].values
x = data.drop(DEPENDENT_VARIABLE,axis=1).values
x = sm.add_constant(x)
features = data.drop(DEPENDENT_VARIABLE,axis=1).columns.tolist()

## Model Fitting
if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] in ["continuous","ordinal"]:
    family = sm.families.Gaussian()
elif DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
    family = sm.families.Binomial()
else:
    raise ValueError("Variable type not recognized")
model = sm.GLM(endog=y, exog=x, family=family)
fit = model.fit(maxiter=1000, full_output=True)
coefs = pd.DataFrame(np.hstack([fit.params.reshape(-1,1), fit.conf_int()]), 
                     index=["intercept"]+features,
                     columns=["coef","lower","upper"]).sort_values("coef").drop("intercept")

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
    plot_coefs = coefs.loc[variable_type_features].copy()
    plot_coefs.index = feature_labels
    fig, ax = shaded_bar_plot(plot_coefs,
                              "coef",
                              xlabel="Coefficient",
                              title=variable_type.replace("_"," ").title())
    fig.savefig(f"{PLOT_DIR}glm_coefficient_{variable_type}.png", dpi=200)
    plt.close(fig)

## Estimate Race and Gender Influence
race_ci = estimate_glm_influence(fit_model=fit,
                                 x=data[features],
                                 dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                 vary_cols=feature_groups["race"],
                                 vary_cols_labels=format_feature_names(feature_groups["race"], "race"),
                                 n_samples=1000,
                                 random_state=42)
gender_ci = estimate_glm_influence(fit_model=fit,
                                   x=data[features],
                                   dv_type=DV_VARIABLE_TYPES[DEPENDENT_VARIABLE],
                                   vary_cols=feature_groups["gender"],
                                   vary_cols_labels=format_feature_names(feature_groups["gender"], "gender"),
                                   n_samples=1000,
                                   random_state=42)

## Cache Causal Effects
gender_ci.to_csv(f"{PLOT_DIR}glm_demographic_influence_gender.csv")
race_ci.to_csv(f"{PLOT_DIR}glm_demographic_influence_race.csv")

## Plot Influence Effects
for ci_df, ci_name in zip([race_ci,gender_ci],["race","gender"]):
    fig, ax = shaded_bar_plot(ci_df,
                              median_col="median",
                              xlabel="Predicted Probability" if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary" else "Predicted {}".format(DEPENDENT_VARIABLE.replace("_"," ").title()),
                              title=f"Demographic Type: {ci_name.title()}")
    fig.savefig(f"{PLOT_DIR}glm_demographic_influence_{ci_name}.png", dpi=200)
    plt.close(fig)

########################
### Graphical Model (Visualization)
########################

## Create Directed Graph (Visualization - Collapse Encodings)
covariates_vis = sorted(set(list(map(lambda c: c if c not in topic_columns else "topic", covariates))))
D_vis = construct_directed_graph(covariates_vis, ["topic"])
D_edges_vis = directed_to_edges(D_vis, covariates_vis+[DEPENDENT_VARIABLE])

## Create Graph
dag_vis = graphs.DAG(vertices=covariates_vis+[DEPENDENT_VARIABLE], di_edges=D_edges_vis)
dag_vis.draw().render(f"{PLOT_DIR}/dag_simple.gv", view=False)

########################
### Graphical Model
########################

print("Modeling Data: Graphical Approach")

## Data Formatting
data = pd.DataFrame(np.hstack([X, Y.reshape(-1,1)]),
                    columns=covariates+[DEPENDENT_VARIABLE])
if DV_VARIABLE_TYPES[DEPENDENT_VARIABLE] == "binary":
    data[DEPENDENT_VARIABLE] = data[DEPENDENT_VARIABLE].astype(int)

## One Hot Encoding
for col in list(data.columns):
    if col == DEPENDENT_VARIABLE:
        continue
    if IV_VARIABLE_TYPES[col] in ["categorical","binary"]:
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

## Create Directed Graph
features = data.drop(DEPENDENT_VARIABLE,axis=1).columns.tolist()
D = construct_directed_graph(features, set(topic_cols_clean))
D_edges = directed_to_edges(D, features+[DEPENDENT_VARIABLE])

## Initial Graphical Model
dag_model = graphs.DAG(features+[DEPENDENT_VARIABLE], D_edges)

## Causal Effect Estimation
gender_ce = estimate_causal_effects(dag_model,
                                    "gender_all",
                                    ["unknown","male","female"],
                                    est_method="aipw",
                                    n_bootstrap=10,
                                    alpha=0.05)
race_ce = estimate_causal_effects(dag_model,
                                  "race_all",
                                  ["unknown","white","black","hispanic","asian","indian","mixed","middle_eastern","pacific_islander"],
                                  est_method="aipw",
                                  n_bootstrap=10,
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
