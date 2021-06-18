
"""
Example showing what happens when you fail to condition
on possible confounds in a graph

True Graph:
D -> T -> Y
D -> Y

True Model:
D ~ error_D
T ~ 0.8 * D + error_T
Y ~ 0.4 * D - 0.5 * T + error_Y
"""

#####################
### Imports
#####################

## External Library
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#####################
### Data Generator
#####################

## Sample Size
N = 1000

## Generate Samples
D = np.random.normal(0, 1, size=N)
T = 0.8 * D + np.random.normal(0, 1, size=N)
Y = 0.4 * D - 0.5 * T + np.random.normal(0, 1, size=N)

## Concatenate Data
df = pd.DataFrame(np.vstack([D, T, Y]).T, columns=["D","T","Y"])

## Fit Models
m1 = sm.GLM.from_formula("Y ~ 1 + D" , data=df).fit()
m2 = sm.GLM.from_formula("Y ~ 1 + D + T", data=df).fit()

## Extract Coefficient
m1_D = m1.params.loc["D"]
m2_D = m2.params.loc["D"]

## Log Results
print("True Coefficient: {:.3f}".format(0.4))
print("Misspecified Coefficient: {:.3f}".format(m1_D))
print("Correctly Specified Coefficient: {:.3f}".format(m2_D))