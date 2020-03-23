import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv('training_data_RF.csv')

# CLEAN
data.drop(data[(data.w>5) | (data.w<0.01)].index, inplace=True)


data.drop(data[data.c>500].index, inplace=True)

X, y = data[['c','theta','A1','n_star']], data[['w']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Our performance metric is the average Absolute Relative Error (ARE)
def mean_ARE(y_pred, y_test):
    y_real = y_test.to_numpy()
    assert len(y_pred)==len(y_real)
    S = 0.
    for i in range(len(y_pred)):
        S += abs(y_pred[i] - y_real[i][0]) / y_real[i][0]
    return (S/len(y_pred))



m = RandomForestRegressor(n_estimators = 100,
       criterion="mae", # Mean-Square Error; or "mae" Mean Absolute Error
       max_depth = None,
       min_samples_split=2,
       min_samples_leaf=1,
       min_weight_fraction_leaf=0.,
       max_features="auto",
       max_leaf_nodes = None,
       min_impurity_decrease=0.,
       min_impurity_split=None,
       bootstrap=True,
       oob_score=False,
       n_jobs=None,
       random_state=None,
       verbose=0)
m.fit(X_train,y_train)
#y_pred = m.predict(X_test)
#print("\n\nTEST SIZE : {data}".format(data = 0.1+size/10))
#print("\n#### average ARE : {d}".format(d = mean_ARE(y_pred,y_test)))

#print("#### explained_variance_score :", explained_variance_score(y_pred,y_test))
#print("\n\n#### r2_score :", r2_score(y_pred,y_test))
#print("\n\n#### mean_absolute_error :", mean_absolute_error(y_pred,y_test))
#print("\n\n#### mean_squared_error :", mean_squared_error(y_pred,y_test))

      
## save model using pickle serializer
pickle.dump(m, open("RF_model.pickle", 'wb'))
