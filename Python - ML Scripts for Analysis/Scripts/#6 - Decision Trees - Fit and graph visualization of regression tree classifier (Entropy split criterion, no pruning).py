# exercise 5.1.3
import numpy as np
from sklearn import tree
import graphviz
# requires data from exercise 5.1.1
from ex5_1_1 import *

# Fit regression tree classifier, deviance (entropy) split criterion, no pruning
# Also, is there a difference between setting min_samples_split=2 or setting it equal to 1.0/N ?
dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2)
#dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=1.0/N)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_deviance.gvz', feature_names=attributeNames)
src=graphviz.Source.from_file('tree_deviance.gvz')
src.render('../tree_deviance', view=True)

print('Ran Exercise 5.1.3')