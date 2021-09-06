import pandas as pd 
import numpy as np 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
column_names = {'T1': 'Temperature(c)','RH_1':'Humidity(%)', 'T2':'Temperature_Living_area', 'RH_2':'Humidity_in_living_area(%)','T3':'Temperature_in_laundry_room','RH_3':'Humidity_in_laundry_room','T4':'Temperature_in_office','RH_4':'Humidity_in_office','T5':'Temperature_in_bathroom','RH_5':'Humidity_in_bathroom','T6':'Temperature_outside_north','RH_6':'Humidity_outside_north','T7':'Temperature_in_ironing_room','RH_7':'Humidity_in_ironing_room', 'T8':'Temperature_in_teenager_room','RH_8':'Humidity_in_teenager_room','T9':'Temperature_in_parents_room','RH_9':'Humidity_in_parents_room','rv1':'Random_variable1','rv2':'Random_variable2'}
df.rename(columns=column_names,inplace=True)
scaler = MinMaxScaler()
df1 = df.drop(columns=['date','lights'])
normalised_df = pd.DataFrame(scaler.fit_transform(df1),columns=df1.columns)
features_df = normalised_df.drop(columns=['Appliances'])
target_variable = normalised_df[['Appliances']]


x_train, x_test, y_train, y_test = train_test_split(features_df, target_variable, test_size = 0.3, random_state = 42)
MultiRegression = LinearRegression()
linear_model.fit(x_train, y_train)
predicted_values = linear_model.predict(x_test)


mae = mean_absolute_error(y_test, predicted_values)
rss = np.sum(np.square(y_test - predicted_values))
meanSqErr = mean_squared_error(y_test, predicted_values)
rmse = np.sqrt(mean_squared_error(y_test,predicted_values))
r2_score = r2_score(y_test, predicted_values)



print('Mean Absolute Error:', mae)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rmse)
print('R2 Score:', r2_score)

rss = np.sum(np.square(y_test - predicted_values))
print(rss)

simpleRegression = df1[['Temperature_Living_area','Temperature_outside_north']].sample(15, random_state=42)
simpleRegression.head(10)

sns.regplot(x ='Temperature_Living_area', y = 'Temperature_outside_north', data = simpleRegression)



x = df1['Temperature_Living_area']
y = df1['Temperature_outside_north']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
pred_value = linear_reg.predict(x_test.values.reshape(-1, 1))

mae = mean_absolute_error(y_test, pred_value)
round(mae, 2)

rmse = np.sqrt(mean_squared_error(y_test, pred_value))
round(rmse, 2)

r2_score = r2_score(y_test, pred_value)
round(r2_score, 2)

rss = np.sum(np.square(y_test - pred_value))
round(rss, 3)
