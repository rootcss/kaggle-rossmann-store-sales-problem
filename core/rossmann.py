#!/usr/bin/python

from xgb_lib import XGBoost
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Pdf")
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

class Rossmann(XGBoost):
  train = None
  test = None
  store = None
  types = None
  features = []
  datasets = {}

  def __init__(self, datasets):
    self.datasets = datasets
    self.types = {
      'CompetitionOpenSinceYear' : np.dtype(int),
        'CompetitionOpenSinceMonth' : np.dtype(int),
        'StateHoliday' : np.dtype(str),
        'Promo2SinceWeek' : np.dtype(int),
        'SchoolHoliday' : np.dtype(float),
        'PromoInterval' : np.dtype(str)
    }

  def loadDataSets(self):
    print("Loading datasets(train|test|store)..")
    self.train = pd.read_csv(self.datasets['train'], parse_dates=[2], dtype=self.types)
    self.test = pd.read_csv(self.datasets['test'], parse_dates=[3], dtype=self.types)
    self.store = pd.read_csv(self.datasets['store'])

  def dataFilterAndCleaning(self):
    print("if store['Open'] is None, set 1..")
    self.train.fillna(1, inplace=True)
    self.test.fillna(1, inplace=True)

    print("if Store is Closed, Ignore!")
    self.train = self.train[self.train["Open"] != 0]
    print("if Sales < 0, Ignore!")
    self.train = self.train[self.train["Sales"] > 0]

    print("Perform a join with store dataset..")
    self.train = pd.merge(self.train, self.store, on='Store')
    self.test = pd.merge(self.test, self.store, on='Store')

  def develop_features(self):
    print("Developing features..")
    self.generate_features(self.features, self.train)
    self.generate_features([], self.test)
    #self.features = list(OrderedDict.fromkeys(self.features))
    print(self.features)
    print('Features have been generated & training data processed.')

  def submission(self, test_probs):
    print("Saving the file at: "+self.datasets['submission_file'])
    result = pd.DataFrame({"Id": self.test["Id"], 'Sales': np.expm1(test_probs)})
    result = result.sort_values(by='Id', ascending=True)
    result.to_csv(self.datasets['submission_file'], index=False)

  def plotting(self, importance, output='plot.png'):
    print("Plotting & Saving the file at: " + output)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    # Kind: bar, barh, area
    featp = df.plot(kind='bar', x='feature', y='fscore', legend=False, figsize=(12, 8), color='Gray')
    plt.title('XGB Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig(output, bbox_inches='tight', pad_inches=1)

  def generate_features(self, features, data):
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    features.extend([
      'Store',
      'CompetitionDistance',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 
        'Promo', 'Promo2',
        'Promo2SinceWeek',
        'Promo2SinceYear'
    ])

    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'month', 'day', 'year'])
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data['day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek


