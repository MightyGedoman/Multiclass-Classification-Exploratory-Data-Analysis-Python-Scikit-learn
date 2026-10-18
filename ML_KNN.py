import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


df = pd.read_csv("data.csv")


X = df[[
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed"
]]

y = df["Type 1"]


zero_not_accepted = ["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]

for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.nan)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.nan, mean)

    
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid = GridSearchCV(
    estimator=pipe,
    param_grid={
        "model__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    },
    cv=3
)

grid.fit(X, y)


results = pd.DataFrame(grid.cv_results_)

print(results[[
    "param_model__n_neighbors",
    "mean_test_score",
    "rank_test_score"
]])

print("Best k:", grid.best_params_)
print("Best score:", grid.best_score_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)


X = df.iloc[:, 5:11]
y = df.iloc[:, 2]

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=15, p=2, metric='euclidean')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

print(cm)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average="weighted"))


results = pd.DataFrame(grid.cv_results_)

plt.figure(figsize=(8,5))

plt.plot(
    results["param_model__n_neighbors"],
    results["mean_test_score"],
    marker="o"
)

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross Validation Accuracy")
plt.title("KNN Accuracy vs Number of Neighbors")
plt.grid(True)

plt.show()


