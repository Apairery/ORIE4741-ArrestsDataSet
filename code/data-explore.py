# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:23:21 2019

@author: liu
"""

import pandas as pd
import numpy as np
import sqlite3 as sql

#%%read data via sqlite3
conn = sql.connect(r".\sqf\sqf.db")
c = conn.cursor()

query = "select * from SQFdata where year >= 2014 and year <= 2016"

data = pd.read_sql(query, conn)
conn.commit()
conn.close()
#%% create the id
data['id'] = range(len(data))
#%% year, month of occurence
yrmn = data.iloc[:, [0, 3, -1]]
yrmn.loc[:, 'datestop'] = (yrmn.loc[:, 'datestop'] / 10e5).astype('int')
#%%suspect personal feature
features = pd.read_excel(r".\SQF-File-Documentation\2016 SQF File Spec.xlsx", header=3)
suspFeat = list(features.loc[80:89, "Variable"])

dataSuspFeat = data.loc[:, suspFeat]
dataSuspFeat = dataSuspFeat.replace(r'^\s*$', np.nan, regex=True)
#dataSuspFeat = dataSuspFeat[dataSuspFeat['age'] != '**']
dataSuspFeat = dataSuspFeat.replace(r'**', np.nan)
dataSuspFeat.age = dataSuspFeat.age.astype('float64')
dataSuspFeat.ht_feet = dataSuspFeat.ht_feet.astype('float64')
dataSuspFeat.ht_inch = dataSuspFeat.ht_inch.astype('float64')
dataSuspFeat['ht'] = dataSuspFeat['ht_feet']*12 + dataSuspFeat['ht_inch']
dataSuspFeat = dataSuspFeat.drop(['ht_feet', 'ht_inch', 'dob'], 1)
dataSuspFeat.weight = dataSuspFeat.weight.astype('float64')
#%% location info
location = list(features.loc[100:106, "Variable"])
dataLoc = data.loc[:, location]
dataLoc = dataLoc.replace(r'^\s*$', np.nan, regex=True)
dataLoc = dataLoc.drop(['city', 'sector', 'state', 'zip', 'post', 'beat'], 1)

#%% other info
other = list(features.loc[41:72, "Variable"])
other += (["explnstp", "othpers", "sumissue"])
dataOther = data.loc[:, other]
dataOther = dataOther.replace('Y', 1)
dataOther = dataOther.replace('N', 0)

dataOther1 = data.loc[:, ["perobs", "perstop", "typeofid"]]
dataOther = dataOther.join(dataOther1, how='left')
#%% merge all features
dataJoin = dataLoc.join(dataSuspFeat, how='left')
dataJoin = dataJoin.join(yrmn, how='left')
dataJoin = dataJoin.join(dataOther, how='left')
