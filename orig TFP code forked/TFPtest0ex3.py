# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, pandas as pd,\
       matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns,\
       arviz as az, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)
mpl.rc("figure", **{"figsize": (8, 6), "autolayout": True})

srrs2 = pd.read_csv("data/srrs2.dat")
srrs2['fips'] = srrs2['stfips']*1000 + srrs2['cntyfips']

cty = pd.read_csv("data/cty.dat")
cty['fips'] = cty['stfips']*1000 + cty['ctfips']

srrs_mn = srrs2[srrs2.state=='MN'].merge(cty[cty.st=='MN'][['fips', 'Uppm']], on='fips').drop_duplicates(subset='idnum')
srrs_mn['county'] = srrs_mn['county'].map(str.strip) # remove blank spaces
u = np.log(srrs_mn['Uppm'].values)
n = len(srrs_mn)

mn_counties = srrs_mn['county'].unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))

county = srrs_mn['county_code'] = srrs_mn['county'].replace(county_lookup).values
radon = srrs_mn['activity'].values
log_radon = srrs_mn['log_radon'] = np.log(radon + .1)
floor_measure = srrs_mn['floor'].values.astype('float')

# Create new variable for mean of floor across counties
xbar = srrs_mn.groupby('county')['floor'].mean().rename(county_lookup).values
