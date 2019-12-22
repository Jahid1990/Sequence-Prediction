## Sequence-Prediction
# Algorithms

RNN, LSTM, bi directional lstm

# Required Packages

import cx_Oracle
import pandas as pd
import keras
import numpy as np
from keras import layers

#Data Loading

con = cx_Oracle.connect(user='jahidulislam', password='database password', dsn='database host name:port/schema name')
df = pd.read_sql_query("select * from tmp_rev_comp_final", con)  
df=df.drop(['ACC'],axis=1)

# Data processing for training

x_train=df.iloc[0:300,9]
y_train=df.iloc[0:300,10:23]
x_test=df.iloc[301:341,9]
y_test=df.iloc[301:341,10:23]
x_train=x_train.to_numpy().reshape(300,1,1)
y_train=y_train.to_numpy().reshape(300,13)
x_test=x_test.to_numpy().reshape(40,1,1)
y_test=y_test.to_numpy().reshape(40,13)

# Model
k = Input(shape=(1,1))
x=LSTM(512, activation='relu',return_sequences=True)(k)
x=LSTM(512, activation='relu',return_sequences=True)(x)
x=LSTM(512, activation='relu',return_sequences=True)(x)
x=LSTM(512, activation='relu',return_sequences=True)(x)
x=LSTM(512, activation='relu',return_sequences=True)(x)
x=LSTM(512, activation='relu',return_sequences=True)(x)
x=LSTM(512, activation='relu',return_sequences=True)(x)
x=LSTM(512, activation='relu',return_sequences=True)(x)
x=LSTM(512, activation='relu',return_sequences=None)(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x)

out=Dense(13,activation='relu')(x)

model=Model(input=k,output=out)

model.summary()

# Prediction
p=model.predict(x_test)
p=pd.DataFrame(p)
print(p.head())
print(pd.DataFrame(y_test).head())


