import os
import traceback
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, FilteredClassifier, \
    PredictionOutput, Kernel, KernelClassifier
from weka.classifiers import Evaluation
from weka.filters import Filter
from weka.core.classes import Random, from_commandline
import weka.plot.classifiers as plot_cls
import weka.plot.graph as plot_graph
import weka.core.types as types

jvm.start()

# access classifier's Java API
labor_file = '../data/labor.arff'
loader = Loader("weka.core.converters.ArffLoader")
labor_data = loader.load_file(labor_file)
labor_data.class_is_last()

jrip = Classifier(classname="weka.classifiers.rules.JRip")
jrip.build_classifier(labor_data)
rset = jrip.jwrapper.getRuleset()
for i in xrange(rset.size()):
    r = rset.get(i)
    print(str(r.toString(labor_data.class_attribute.jobject)))


prism = Classifier(classname="weka.classifiers.rules.DecisionTable", options=["-R"])
prism.build_classifier(labor_data)
print prism.jwrapper.toString()
# print prism.jwrapper.m_dtInstances

j48 = Classifier(classname="weka.classifiers.trees.J48")
j48.set_property("confidenceFactor", types.double_to_float(0.3))
j48.build_classifier(labor_data)
print(j48)
print(j48.graph)

jvm.stop()