import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv('training_data_RF.csv')

# CLEAN
data.drop(data[(data.w>5) | (data.w<0.01) | (data.n<10)].index, inplace=True)


# Our performance metric is the average Absolute Relative Error (ARE)
# Entry variables are DataFrames
def mean_ARE(y_pred, y_real):
    assert len(y_pred)==len(y_real)
    S = 0.
    for i in range(len(y_pred)):
        S += abs(y_pred[i] - y_real[i][0]) / y_real[i][0]
    return (S/len(y_pred))


res = pd.DataFrame(columns = ['perc', 'size', 'mean_are', 'explnd_var','r2','mae','mse'])

for perc in range(2, 11):
    sub_data = data.sample(frac = perc/10)
    X, y = sub_data[['c','theta','A1','n_star']], sub_data[['w']]
    for siz in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1+siz/10)
        m = RandomForestRegressor(n_estimators = 200,
               criterion="mae", # Mean-Square Error; or "mae" Mean Absolute Error
               max_depth = None)
        m.fit(X_train, y_train.w.ravel())
        y_pred = m.predict(X_test)
        #print("\n#### average ARE : {d}".format(d = mean_ARE(y_pred,y_test)))
        # MEAN ARE
        Idx = y_test.index
        y_pred = pd.DataFrame(y_pred, index=Idx, columns=['w_pred'])
        data.loc[Idx, 'N_pred'] = round(data.loc[Idx, 'n'] + y_pred.w_pred * data.loc[Idx, 'A1'] / (1-data.loc[Idx, 'n_star'])).astype(int)
        data.loc[Idx, 'are'] = abs(data.loc[Idx, 'N_real'] - data.loc[Idx, 'N_pred']) / data.loc[Idx, 'N_real']
        
        res.loc[str(perc)+"-"+str(siz)] = {'perc':perc, 'size':siz,
                'mean_are': data.are.mean(),
                'explnd_var':explained_variance_score(y_pred,y_test),
                'r2':r2_score(y_pred,y_test),
                'mae':mean_absolute_error(y_pred,y_test),
                'mse':mean_squared_error(y_pred,y_test)}
        #print("\n\n#### explained_variance_score :", explained_variance_score(y_pred,y_test))
        #print("\n\n#### r2_score :", r2_score(y_pred,y_test))
        #print("\n\n#### mean_absolute_error :", mean_absolute_error(y_pred,y_test))
        #print("\n\n#### mean_squared_error :", mean_squared_error(y_pred,y_test))
print(res)
      
## save model using pickle serializer
#pickle.dump(m, open("RF_model.pickle", 'wb'))
