import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

from sklearn.externals.six import StringIO
import pydotplus

iris = load_iris()

# print iris.feature_names
# print iris.target_names

# print iris.data[0]
# print iris.target_names[iris.target[0]]
# print iris.target_names[iris.target[1]]

# for i in range(len(iris.target)):
#     print "Example %d: label %s, feature %s" % (i, iris.target_names[iris.target[i]], iris.data[i])

test_ids = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_ids)
train_data = np.delete(iris.data, test_ids, axis=0)

#testing data
test_target = iris.target[test_ids]
test_data = iris.data[test_ids]

classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)

print test_target
print classifier.predict(test_data)

dot_data = StringIO()
tree.export_graphviz(classifier,
                                out_file=dot_data,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True,
                                rounded=True,
                                impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("mjau.pdf")

print test_data[1], test_target[1]