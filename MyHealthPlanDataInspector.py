import pandas.rpy.common as com
from prettypandas import PrettyPandas
filename = "data/HealthPlan/Beweging module 1.sav"
w = com.robj.r('foreign::read.spss("%s", to.data.frame=TRUE)' % filename)
w = com.convert_robj(w)
print list(w.columns)
# print w.describe()
table = PrettyPandas(w.head(5))
table