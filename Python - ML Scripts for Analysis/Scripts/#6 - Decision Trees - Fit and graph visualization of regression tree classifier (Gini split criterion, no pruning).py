# exercise 5.1.2
import numpy as np
from sklearn import tree
import graphviz

# requires data from exercise 5.1.1
from ex5_1_1 import *

# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)
#graphviz.render('dot','png','tree_gini',quiet=False)
src=graphviz.Source.from_file('tree_gini.gvz')
## Comment in to automatically open pdf
## Note. If you get an error (e.g. exit status 1), try closing the pdf file/viewer
src.render('../tree_gini', view=True)

print('Ran Exercise 5.1.2')