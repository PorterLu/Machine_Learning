import copy
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import neural_network
from sklearn import ensemble

#读取数据
data = pd.read_csv("C:/traindata.txt", sep=" ",header=None,encoding="utf-8")
label = pd.read_csv("C:/trainlabel.txt", sep=" ", header=None,encoding="utf-8")
test = pd.read_csv("C:/testdata.txt", sep=" ", header=None,encoding="utf-8")
data = data.values
label = label.values

#imbalance处理
tmp_data = data
tmp_label = label
for i in range(600):
    if label[i][0] == 1:
        data = np.row_stack((data, copy.deepcopy(data[i])))
        label = np.row_stack((label, [1]))

#KNN
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(data, label)
knn.score(tmp_data, tmp_label)

#DecisionTree
dt= DecisionTreeClassifier()
dt.fit(data, label)
dt.score(tmp_data, tmp_label)

#svm
svm = SVC()
svm.fit(data, label)
svm.score(tmp_data, tmp_label)

#neural network
mlp = neural_network.MLPClassifier(hidden_layer_sizes=(1000), activation="logistic",
                                  solver="adam", alpha=0.001,
                                  batch_size=20, learning_rate="constant",
                                  learning_rate_init=0.001, power_t=0.5, max_iter=50, tol=1e-4)
mlp.fit(data, label)
mlp.score(tmp_data, tmp_label)

#ensemble learning
bagging = ensemble.BaggingClassifier(KNeighborsClassifier(),
                                    max_samples=800, max_features=0.4,oob_score=True,n_jobs=-1)
bagging.fit(data, label)
bagging.score(tmp_data, tmp_label)


#输出预测结果
bagging.predict(test)