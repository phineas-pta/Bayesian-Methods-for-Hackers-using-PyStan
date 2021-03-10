# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, pandas as pd,\
       matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns,\
       arviz as az, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)

#%% data

srrs2 = pd.read_csv("data/srrs2.dat")
srrs2['fips'] = srrs2['stfips']*1000 + srrs2['cntyfips']

cty = pd.read_csv("data/cty.dat")
cty['fips'] = cty['stfips']*1000 + cty['ctfips']

srrs_mn = srrs2[srrs2.state=='MN'].merge(cty[cty.st=='MN'][['fips', 'Uppm']], on='fips').drop_duplicates(subset='idnum')
srrs_mn['county'] = srrs_mn['county'].map(str.strip) # remove blank spaces

mn_counties = srrs_mn['county'].unique()
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
radon = srrs_mn['activity'].values

county = srrs_mn['county'].replace(county_lookup).values
N = len(srrs_mn)
log_radon = np.log(radon + .1) # +0.1 to make log scale
floor_measure = srrs_mn['floor'].values.astype('float')
counties = len(mn_counties)
u = np.log(srrs_mn['Uppm'].values)
xbar = srrs_mn.groupby('county')['floor'].mean().rename(county_lookup).values
x_mean = xbar[county]
county += 1 # Stan is 1-based index

# +0.1 to make log scale
sns.displot(log_radon, bins = "sqrt", kde = True)
sns.displot(radon + .1, bins = "sqrt", kde = True, log_scale = True)

#%% fittinh

@tfd.JointDistributionCoroutineAutoBatched
def model():
	uranium_weight = yield tfd.Normal(loc=0., scale=1., name='uranium_weight')
	county_floor_weight = yield tfd.Normal(loc=0., scale=1., name='county_floor_weight')
	county_effect = yield tfd.Sample(
		tfd.Normal(loc=0., scale=county_effect_scale),
		sample_shape=[counties], name='county_effect'
	)
	yield tfd.Normal(
		loc=log_uranium * uranium_weight +\
		    floor_of_house * floor_weight +\
		    floor_by_county * county_floor_weight +\
		    tf.gather(county_effect, county, axis=-1) +\
		    bias,
		scale=log_radon_scale[..., tf.newaxis],
		name='log_radon'
	)
