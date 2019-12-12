import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Create feature matrix with floor area and lease price
X = np.array([[10000,45],
              [13000,52],
              [11000,52],
              [18000,55],
              [16000,57],
              [13000,62],
              [12000,62]])

# Create feature matrix with missing values(lease price)
X_with_nan = np.array([[np.nan, 54]])
# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])
# Predict missing values' class
imputed_values = trained_model.predict(X_with_nan[:,1:])

# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))

# Join two feature matrices
ans = np.vstack((X_with_imputed, X))
print(ans)

print("Missing value for",ans[0][1]," is $",ans[0][0])