import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('training_data_RF.csv')
# CLEAN
data.drop(data[(data.w>5) | (data.w<0.005) | (data.n<5)].index, inplace=True)

# FIT
X, y = data[['c','theta','A1','n_star']], data[['w']]
test_size = round((len(data)-500)/len(data),3) # 500 training data exactly for memory volume purpose
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
m = RandomForestRegressor(n_estimators = 100,
                          criterion="mse", # "mse" Mean-Square Error; or "mae" Mean Absolute Error
                          max_depth = None)
m.fit(X_train, y_train.w.ravel())
y_pred = m.predict(X_test)


# Our performance metric is the average Absolute Relative Error (ARE)
Idx = y_test.index
y_test.loc[Idx, 'w_pred'] = pd.Series(data=y_pred, index=y_test.index)
y_test.loc[Idx, 'N_rf'] = round(data.loc[Idx, 'n'] + y_test.w_pred * data.loc[Idx, 'A1'] / (1-data.loc[Idx, 'n_star'])).astype(int)
y_test.loc[Idx, 'N_real'] = data.loc[Idx, 'N_real']
y_test.loc[Idx, 'are'] = abs(y_test.N_real - y_test.N_rf) / y_test.N_real

# Performances atteintes : average ARE = 0.4
# c'est-à-dire la prédiction est éloignée de 40% de la valeur de popularité réelle

# Plot mean ARE for each decile
y_test.loc[Idx, 'deciles'] = pd.qcut(y_test.N_real, q=10)

box_labels = pd.DataFrame(y_test.deciles.unique(), columns=['interval'])
box_labels['sort_key']=box_labels.interval.map(lambda x : x.left)
box_labels.sort_values(by='sort_key', inplace=True)
box_labels.drop(['sort_key'], axis=1, inplace=True)

box_data = [y_test[y_test.deciles==d].are.to_numpy() for d in box_labels.interval]
box = plt.boxplot(box_data,
            sym='', # hide "outliers"
            vert=True, # vertical box alignment
            patch_artist=True, # fill with color
            labels=[str(d) for d in box_labels.interval],
            showmeans=True,
            meanline=True,
            meanprops=dict(linestyle='--', linewidth=2.5, color='r'))
plt.title("Average ARE per deciles (cascades sorted by real number of retweets)\n\n" + \
          "Total average ARE = {a}% for {c} tested data ( model trained with {b} data)\n".format(
                  a=round(100*y_test.are.mean(),2), b=len(y_train), c=len(y_test)))
for line in box['medians']:
    x, y = line.get_xydata()[1]
    plt.text(x, y, '%s' % round(y,3), horizontalalignment='right', verticalalignment='bottom')
for line in box['means']:
    x, y = line.get_xydata()[1] # bottom of left line
    plt.text(x, y, '%s' % round(y,3), horizontalalignment='right', verticalalignment='bottom')
plt.show()
#