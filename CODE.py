#!/usr/bin/env python
# coding: utf-8

# In[1]:


## MANIK MARWAHA
## A1797063
## PROJECT 2
## Restaurant Visitor Forecasting

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[2]:


## Reading datasets into a dictionary

files = {
    'visits': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\air_visit_data.csv", encoding="UTF-8"),
    'store': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\air_store_info.csv", encoding="UTF-8"),
    'hs': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\hpg_store_info.csv", encoding="UTF-8"),
    'reserve': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\air_reserve.csv", encoding="UTF-8"),
    'hr': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\hpg_reserve.csv", encoding="UTF-8"),
    'id': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\store_id_relation.csv", encoding="UTF-8"),
    '_test': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\sample_submission.csv", encoding="UTF-8"),
    'hol': pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\data\date_info.csv", encoding="UTF-8").rename(columns={'calendar_date':'visit_date'})
    }


# In[3]:


files['visits'].isnull().sum() ## visitors data from air system


# In[4]:


files['store'].isnull().sum()  ## information about restaurants from air system


# In[5]:


files['hs'].isnull().sum() ## information about restaurants from hpg system


# In[6]:


files['reserve'].isnull().sum()  ## reservation information of air restaurants


# In[7]:


files['hr'].isnull().sum()  ## resevations made in hpg system


# In[8]:


files['id'].isnull().sum()  ## helps to merge restaurant from both systems


# In[9]:


files['_test'].isnull().sum()  ## sample submission


# In[10]:


files['hol'].isnull().sum()  ## information about dates


# In[11]:


print("unique AIR restaurants:-",len(files['visits'].air_store_id.unique()))
print("Total number of AIR restaurant's locations",len(files['store'].air_area_name.unique()))
print("Average number of daily visitors:-",files['visits'].visitors.mean())
print("unique genre in restaurants in AIR system:-",len(files['store'].air_genre_name.unique()))
print("Training data duration:-{} to {}\n\n\n".format(files['visits'].visit_date.min(),files['visits'].visit_date.max()))


# **EXPLORATORY DATA ANALYSIS**
# 
# 

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn


# In[13]:


## Total visitors each day
files['visits'].visit_date = pd.to_datetime(files['visits'].visit_date)
visitors_everyday = files['visits'].groupby('visit_date').sum()
visitors_everyday.head(3)


# In[14]:


ax = visitors_everyday.plot(figsize=[15,6], c='green')


# In[15]:


files['visits'].head()


# In[16]:


## finding the total number of visitors on each day of the week

files['visits']['weekday'] = files['visits'].visit_date.dt.weekday
visitors_day_of_the_week = files['visits'].groupby('weekday')['visitors'].sum()
ax = visitors_day_of_the_week.plot.bar(figsize=[10,5])
ax.set_xlabel('(Monday=0, Sunday=6)')


# In[17]:


files['visits'].head()


# In[18]:


files['visits']=files['visits'].drop(['weekday'], axis=1)


# In[19]:


files['visits'].head()


# In[20]:


files['reserve'].head()


# In[21]:


## findng what time maximum reservations are made
files['reserve'].visit_datetime = pd.to_datetime(files['reserve'].visit_datetime)
files['reserve'].visit_datetime.dt.hour.value_counts().sort_index().plot.bar(figsize=[12,6],title='AIR SYSTEMS')


# In[22]:


## findng what time maximum reservations are made from the hpg system
files['hr'].visit_datetime = pd.to_datetime(files['hr'].visit_datetime)
files['hr'].visit_datetime.dt.hour.value_counts().sort_index().plot.bar(figsize=[12,6],title='HPG SYSTEMS')


# **from the above 2 graphs it is clear maximum number of people visit for the dinner**

# In[23]:


files['reserve'].reserve_datetime = pd.to_datetime(files['reserve'].reserve_datetime)
files['reserve']['reserve_ahead'] = files['reserve'].visit_datetime - files['reserve'].reserve_datetime
files['reserve']['hours_ahead'] = files['reserve'].reserve_ahead / pd.Timedelta('1 hour')
ax = files['reserve'].hours_ahead.hist(figsize=[15,6],bins=10000)
ax.set_xlim([0,500])
ax.set_xticks(np.arange(0, 500, 24))


# In[24]:


files['hr'].reserve_datetime = pd.to_datetime(files['hr'].reserve_datetime)
files['hr']['reserve_ahead'] = files['hr'].visit_datetime - files['hr'].reserve_datetime
files['hr']['hours_ahead'] = files['hr'].reserve_ahead / pd.Timedelta('1 hour')
ax = files['hr'].hours_ahead.hist(figsize=[15,6],bins=10000)
ax.set_xlim([0,500])
ax.set_xticks(np.arange(0, 500, 24))


# In[25]:


files['reserve']=files['reserve'].drop(['reserve_ahead','hours_ahead'],axis=1)
files['hr']=files['hr'].drop(['reserve_ahead','hours_ahead'],axis=1)


# **DATA PRE-PROCESSING**

# In[26]:


## merging hpg_reserve dataset and id dataset
files['hr'] = pd.merge(files['hr'], files['id'], how='inner', on=['hpg_store_id'])

## reservation datasets of both air and hpg systems converting to date format by removing the time
## also finding difference between reservation and visiting datetime
for df in ['reserve','hr']:
    files[df]['visit_datetime'] = pd.to_datetime(files[df]['visit_datetime'])
    files[df]['visit_datetime'] = files[df]['visit_datetime'].dt.date
    files[df]['reserve_datetime'] = pd.to_datetime(files[df]['reserve_datetime'])
    files[df]['reserve_datetime'] = files[df]['reserve_datetime'].dt.date
    files[df]['reserve_datetime_diff'] = files[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    
    
    
    ## columns rename
    files[df] = files[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})
    print(files[df].head(3))


# In[27]:


files['_test'].head(3)


# In[28]:


# Extracting datetime features by using historical data of total visits from air system
files['visits']['visit_date'] = pd.to_datetime(files['visits']['visit_date'])
files['visits']['dow'] = files['visits']['visit_date'].dt.dayofweek
files['visits']['year'] = files['visits']['visit_date'].dt.year
files['visits']['month'] = files['visits']['visit_date'].dt.month
files['visits']['visit_date'] = files['visits']['visit_date'].dt.date

## extracting datetime features of sample submissiion
## split id column into - gives visit date and id of the restaurant
files['_test']['visit_date'] = files['_test']['id'].map(lambda x: str(x).split('_')[2])
files['_test']['air_store_id'] = files['_test']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
files['_test']['visit_date'] = pd.to_datetime(files['_test']['visit_date'])
files['_test']['dow'] = files['_test']['visit_date'].dt.dayofweek
files['_test']['year'] = files['_test']['visit_date'].dt.year
files['_test']['month'] = files['_test']['visit_date'].dt.month
files['_test']['visit_date'] = files['_test']['visit_date'].dt.date
## HENCE WE MODIFIED THE SAMPLE SUBMISSION DATASET


# In[29]:


print("Total unique restaurants in testing dataset:-",len(files['_test'].air_store_id.unique()))
print("Test data duration:-{} to {}".format(files['_test'].visit_date.min(),files['_test'].visit_date.max()))


# In[30]:


unique_stores = files['_test']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)


# In[31]:


# Creating statistical features from restaurant visits every day of the week
# Minimun number of visitors per store
tmp = files['visits'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

# Mean number of visitors per store
tmp = files['visits'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

# Median number of visitors per store
tmp = files['visits'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

# Maximum number of visitors per store
tmp = files['visits'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

# Visitor Count per store
tmp = files['visits'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])


# In[32]:


stores = pd.merge(stores, files['store'], how='left', on=['air_store_id'])
 
encoder = LabelEncoder()
stores['air_genre_name'] = encoder.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = encoder.fit_transform(stores['air_area_name'])

files['hol']['visit_date'] = pd.to_datetime(files['hol']['visit_date'])
files['hol']['day_of_week'] = encoder.fit_transform(files['hol']['day_of_week'])
files['hol']['visit_date'] = files['hol']['visit_date'].dt.date

files['visits'] = pd.merge(files['visits'], files['hol'], how='left', on=['visit_date'])
files['_test'] = pd.merge(files['_test'], files['hol'], how='left', on=['visit_date'])

# Create train and test dataframes
train = pd.merge(files['visits'], stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(files['_test'], stores, how='left', on=['air_store_id','dow'])

for df in ['reserve','hr']:
    train = pd.merge(train, files[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, files[df], how='left', on=['air_store_id','visit_date'])

col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]

# Impute missing values in case any
train = train.fillna(-1)
test = test.fillna(-1)

def coerce_types():
    for c, dtype in zip(train.columns, train.dtypes):
        if dtype == np.float64:
            train[c] = train[c].astype(np.float32)
        
    for c, dtype in zip(test.columns, test.dtypes):
        if dtype == np.float64:
            test[c] = test[c].astype(np.float32)


# In[33]:


# Train and Test feature selection
# Drop store_id and visit date as they won't help the model
# visit_date was used to extract date-time features so its not needed now.
# Visitors is our target variable so we can't train on it.

train_features = train.drop(['air_store_id','visit_date','visitors'], axis=1)
train_target = np.log1p(train['visitors'].values)
print(f"Train_features_size: {train_features.shape}, Target_size: {train_target.shape}")

X_test = test.drop(['id','air_store_id','visit_date','visitors'], axis=1)


# In[34]:


# Root Mean Squared logarithemic Error as evaluation metric
from sklearn.metrics import make_scorer
def rmsle(y,y_predicted): 
    
    y = np.expm1(y)
    y_predicted = np.expm1(y_predicted)
    
    return np.sqrt(np.square(np.log(y_predicted + 1) - np.log(y + 1)).mean())

# scoring function to be used in parameter tuning
score = make_scorer(rmsle, greater_is_better=False)


# In[35]:


parameters = {'learning_rate':[0.1,0.01,0.001],
              'n_estimators':[50,100,150,200],
              'min_child_weight':[0.8,0.9,1],
              'subsample':[0.5,0.6,0.7],
              'colsample_bytree':[0.3,0.4,0.5,0.6],
              'max_depth': [2,4,8],
              'reg_alpha': [0,0.2,0.4,0.6]}



model = xgb.XGBRegressor(objective='reg:squarederror')

# cross validation
model_cv = GridSearchCV(estimator=model,
                     param_grid=parameters,
                     cv=5,
                     return_train_score=True,
                     n_jobs=-1,
                     scoring=score)
model_cv.fit(train_features,train_target)

model_cv.best_params_


# In[36]:


def build_model():
    """Visitor predicition model.
    Returns:
         Model
    """
    model = xgb.XGBRegressor(max_depth=8,
                              n_estimators=150,
                              objective='reg:squarederror',
                              colsample_bytree=0.5,
                              metric='auc',
                              min_child_weight=0.8,
                              subsample=0.6,
                              nthread=2,
                              learning_rate=0.1,
                              random_state=77,
                             reg_alpha=0.2
                              )
    return model

## since we already took the log of train_target so this function will give us RMSLE
def rmse(y_true, y_pred):
    
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_model(train_files, labels, model):
    
    ## trains visitor prediction model
    folds = TimeSeriesSplit(n_splits=8)
    scores = []
    preds=[]

    for i, (train_index, val_index) in enumerate(folds.split(train_files, labels), 1):
        ## training and validation subsets
        X_train, y_train = train_files[train_index], np.take(labels, train_index, axis=0)
        X_val, y_val = train_files[val_index], np.take(labels, val_index, axis=0)
        
        ## fitting the model
        model.fit(X_train, y_train)
        
        ## calculating rmsle for training and testing subsets for each fold
        t_score = rmse(y_train, model.predict(X_train))
        val_preds = model.predict(X_val)
        score = rmse(y_val, val_preds)
        scores.append(score)
        preds.append(val_preds)
        print(f'Fold-{i}: Train_RMSLE: {t_score}, Validation_RMSLE: {score}')
    print(f'Mean_RMSLE of validation set: {np.round(np.mean(scores), 4)}')
    print(f'Normalized_RMSLE using Standard Deviation:{np.mean(scores)/np.std(preds)}')
    print('\n===============Finished Training====================\n')
    
    return model

def predict_on_test(test_files, test_features, model):
    ## Run predictions on test files
    ## take anti log as we took the log before
    test_predictions = np.expm1(model.predict(test_features))
    test_files['visitors'] = test_predictions

    test_files[['id','visitors']].to_csv('submission.csv', index=False)


if __name__=='__main__':
  coerce_types()
  raw_model = build_model()
  print('====================Training Model===========================')
  trained_model = train_model(train_features.values, train_target, raw_model)
  print('====================Test Prediction===========================')
  predict_on_test(test, X_test.values, trained_model)






