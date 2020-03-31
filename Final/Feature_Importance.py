import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def Feature_Importance():

    data = pd.read_csv("/Users/ange/Downloads/Training-Data-Sets.csv", header=None)
    X = data.iloc[1:12000,2:38]
    y = data.iloc[1:12000,1] 

    model = RandomForestClassifier(max_depth=15, random_state=0)
    model.fit(X, y)
    feature=model.feature_importances_

    for i in feature:
        print(format(i, 'f'))
    
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(feature)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], feature[indices[f]]))    
    
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), feature[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
Feature_Importance()