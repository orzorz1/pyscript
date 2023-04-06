import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# 加载iris数据集
data = datasets.load_iris()
# iris数据集的标签
iris_target = data.target
# 利用Pandas将iris数据集的特征转化为DataFrame格式
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
# 将数据集划分为两部分，训练集80%，测试集20%
X_train, X_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size=0.2, random_state=0)
# 实例化高斯朴素贝叶斯分类器
clf = GaussianNB()
# 训练分类器
clf.fit(X_train, y_train)
# 评估
test_predict = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test,test_predict)
print('The accuracy of the NB for Test Set is: %d%%' % (accuracy*100))
print(test_predict)
print(y_test)

# 预测
y_proba = clf.predict_proba(X_test[:1])
print(X_test[:1])
print(clf.predict(X_test[:1]))
print("预计的概率值:", y_proba)
