# exercise 5.1.6
import numpy as np
from sklearn import tree
import graphviz
# requires data from exercise 5.1.4
from ex5_1_5 import *

# Fit regression tree classifier, Gini split criterion, pruning enabled
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=100)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini_Wine_data.gvz', feature_names=attributeNames)
src=graphviz.Source.from_file('tree_gini_Wine_data.gvz')
src.render('../tree_gini_Wine_data', view=True) 
print('Ran Exercise 5.1.6')