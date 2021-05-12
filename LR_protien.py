# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:34:13 2021

@author: dell
"""

#############################################################################3##
#                 1st AlGO -->Linear regression
#
###############################################################################


#------------------------------------------------------------------------------
#                         STEP 1 -->Import libraries
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,KFold,cross_val_score

import scipy.stats as stats

#VIF-->variable infliation factor -->used to check score of multicolinarity
from statsmodels.stats.outliers_influence import variance_inflation_factor

#feature selection for regression 
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import seaborn as sns
# OLS library for linear regression
import statsmodels.api as sm
#-----------------------------------------------------------------------------
#                       STEP 2--> Read the dataset
#-----------------------------------------------------------------------------
prot=pd.read_csv('prot.csv')
prot.info
prot.describe()
prot.head(3)
prot.columns
prot.shape
#to change display settings to get full view
#pd.set_option('display.expand_frame_repr',False)

#----------------------------------------------------------------------------
#                       STEP 3 --> EDA
#---------------------------------------------------------------------------
#Remove the unwanted columns (drop that cols)

prot=prot.drop(['Unnamed: 0','X','X.1','s','t','u'],axis=1)
prot.columns
prot.shape

#Dimensions(Rows,Columns)

prot.shape[0]
prot.shape[1]

#head/tail
prot.head(2)
prot.head(2)
#Remove the columns which have singularity
#singularity means one term repeated more than 85 %
#ssp columns have 100% 0s are there so remove it

prot=prot.drop(['ssp'],axis=1)
prot.shape[1]
#so we got final 9 columns
#------------------------------------------------------------------
#check for nulls
#------------------------------------------------------------------
prot.isnull().sum()#so here we got 28 nulls in ('npe_area')
#to check exact nulls position
#prot[['npe_area']][prot.npe_area.isnull()]
#other way to check

ndx=prot.npe_area[prot.npe_area.isnull()].index
ndx
#imputing nulls by refrencing the total surafec area
prot[['tot_surf_area','npe_area']][prot.npe_area.isnull()]
#here our logic to impute the nulls and zeros area total surface area is 3times more than npe_area
#by observation total surface area= 3 X npe arae
#on that basis
#imputing
prot.npe_area[prot.npe_area.isnull()]=prot.tot_surf_area/3
#to check index that imputed
prot[['tot_surf_area','npe_area']][prot.index.isin(ndx)]
#nulls are imputed manually by maths
#here our logic to impute the nulls and zeros area total surface area is 3times more than npe_area
#by observation total surface area= 3 X npe arae
prot[['tot_surf_area','npe_area']]
#----------------------------------------------------------------------
#chekcking for zero
#---------------------------------------------------------------------
prot[prot==0].count()

#select * where tot_surf_area=0
prot[['tot_surf_area','npe_area']][prot.tot_surf_area==0]
#so we got 20 0S in total sur area
#imput it by multiplyng npe area by 3

ndx=prot[['tot_surf_area','npe_area']][prot.tot_surf_area==0].index
ndx#so here we got 20 index that have 0 s 

#imputation of zero 

prot.tot_surf_area[prot.tot_surf_area==0]=prot.npe_area*3

#verify the change

prot[['tot_surf_area','npe_area']][prot.index.isin(ndx)]

#check for 0
prot[prot==0].count(0)

#so now ther is 90 counts on RMSD column
#fin realtion between two nearby columns

x=prot.tot_surf_area
y=prot.RMSD
plt.scatter(x,y)
plt.xlabel('Total area')
plt.ylabel('RMSD')
#so there is no any corealtion in between two cols
## check for 0
prot[prot==0].count()

prot[['RMSD','fa_enppr']][prot.RMSD > 0].head(50).sort_values('RMSD')

# since the Y-var has 0, drop these records
ndx = prot[prot.RMSD==0].index
len(ndx)

print("before dropping rows, shape = ", prot.shape)
prot = prot.drop(ndx,axis=0)
print("after dropping rows, shape = ", prot.shape)

prot[prot==0].count()


# check the distibution and outliers of features
cols = list(prot.columns)
cols.remove('RMSD')

len(cols)
#-------------------------------------------------------------
# outliers
#----------------------------------------------------------
prot.boxplot('tot_surf_area',vert=False)

nrow=4; ncol=2; npos=1
#whis=<n>idicates the IQR formula .can override the default 1.5 to
#any value supress to outleirs from apearing in boxplot
fig = plt.figure()
for c in cols:
    fig.add_subplot(nrow,ncol,npos)
    prot.boxplot(c,vert=False,whis=5)
    npos+=1
#so here is outleirs beacuse of there range so #transform the columns

#-----------------------------------------------------------------
#               Displot
#---------------------------------------------------------------

nrow=4;ncol=2;npos=1
fig=plt.figure()
for c in cols:
    fig.add_subplot(nrow,ncol,npos)
    sns.distplot(prot[c])
    npos+=1

#------------------------------------------------------------------
# test on the model 
#-----------------------------------------------------------------

#agistino-Pearsons test for normality
#H0:normal distributed
#H1:not normally distributed

from scipy.stats import normaltest
aptest={}

for c in cols:
    tstat,pval=normaltest(prot[c])
    if pval<0.05:
        aptest[c]='Not normally distributed'
    else:
        aptest[c]='Normally distributed'
        
aptest
#so all the columns are not normally distributed
#-----------------------------------------------------------------
                    #correlation matrix
#------------------------------------------------------------------
#only lower traingle

cor=prot[cols].corr()
cor=np.tril(cor)
sns.heatmap(cor,xticklabels=cols,yticklabels=cols,vmin=-1,vmax=1,annot=True,square=False)


#based on the matrix there are some correlated variables thta needs to be removed

#for linear regression ,data type have to be numerc
prot.dtypes


########################################################################
#                        STEP 5--> Spitting of data
#######################################################################

#split the data into train/test
#trainx/trainy/testx/testy

trainx,testx,trainy,testy=train_test_split(prot.drop('RMSD',axis=1)
                                           ,prot['RMSD']
                                           ,test_size=0.3)
print('trainx={}, trainy={},testx={},testy={}'.format(trainx.shape,trainy.shape,testx.shape,testy.shape))

#################################################################################
#               STEP 6 -->Model Building
##################################################################################

#regression model building using

#TO add a cosntant term to the trainx and testx
#this will ensure that the model summary has the 'intercept' term displyed
trainx=sm.add_constant(trainx)
testx=sm.add_constant(testx)

#OLS-ordinary least square method
m1=sm.OLS(trainy,trainx).fit()

#summarise the model
m1.summary()

p1=m1.predict(testx)
p1.head(15)
p1[0:5]
df=pd.DataFrame('actual':testy,'predicted':p1)

mse1=round(mean_squared_error(testy,p1),3)
mse1




################################################################################
#                           All other REGRESSORS
#########################################################################

#2)Decision tree regressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
#DT model

m2=DecisionTreeRegressor(criterion='mse',max_depth=5).fit(trainx,trainy)
p2=m2.predict(testx)
p2[0:5]
p2[3779]

mse2=round(mean_squared_error(testy,p2),3)
mse2


#3)Random forest regression

from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
m3=RandomForestRegressor().fit(trainx,trainy)
p3=m3.predict(testx)
mse3=round(mean_squared_error(testy,p3),2)
mse3

#$4)KNN regression
trainx_std=trainx.copy()
testx_std=testx.copy()

minmax=preprocessing.MinMaxScaler()

#scale the train data
sc=minmax.fit_transform(trainx_std.iloc[:,:])
trainx_std.iloc[:,:]=sc

#scale the test data

sc=minmax.fit_transform(testx_std.iloc[:,:])
testx_std.iloc[:,:]=sc

trainx_std.head(5)
testx_std.tail(5)


#for regression there is not requirement of odd number of classifier
#so range is 3-12 include even numbers
from sklearn import neighbors
nn=range(3,12)
list(nn)


mse_cv=[]

for k in nn:
    m=neighbors.KNeighborsRegressor(n_neighbors=k).fit(trainx_std,trainy)
    err=cross_val_score(m,trainx_std,trainy,cv=5,scoring='neg_mean_squared_error')
    err=np.round(np.mean(err),3)
    mse_cv.append(err)
    

#print MSE score are all in negaative ,so cinvert it into +ve
print(mse_cv)

#convert aall negative to positive using LAMBDA function

mse_cv=list(map(lambda x: abs(x),mse_cv))
mse_cv
#select select minimum MSE score and its coresponding K value

min(mse_cv)

bestk=nn[mse_cv.index(min(mse_cv))]
bestk

#build the model now with best k
from sklearn.metrics import mean_squared_error

m4=neighbors.KNeighborsRegressor(n_neighbors=bestk).fit(trainx_std,trainy)
p4=m4.predict(testx_std)
mse4=mean_squared_error(testy,p4)
mse4


#SVM Reggression

from sklearn import svm,preprocessing
#we already scaled the data we can use directly kernels now

#so kernels are
#1)linear
#2)plynomial
#3)RBF
#4)sigmoid kernel

#test to check the best kernel for it
# since data is already scaled, we can directly use the kernels on the scaled data

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernels

# determine the R-square for each regression model, for each Kernel
for k in kernels:
    m = svm.SVR(kernel = k).fit(trainx_std,trainy)
    rsq = m.score(testx_std,testy)
    print("Kernel = {}, R-Square = {}".format(k, rsq))

# based on the R-square, it is clear that Sigmoid kernel does not fit the data properly. 


# SVM regression
def svmRegression(ker,trainx,trainy,testx,testy,bestc=1,bestg='scale'):
    model=svm.SVR(kernel=ker,C=bestc,gamma=bestg).fit(trainx,trainy)
    pred = model.predict(testx)
    mse = mean_squared_error(testy,pred)
    return(pred, mse)

m_mse=[]; p5=[]

# run the regression model for each Kernel
for k in kernels:
    pred,mseval=svmRegression(k,trainx_std,trainy,testx_std,testy)  
    p5.append(pred)
    m_mse.append(round(mseval,3))



# create a dataframe to store the MSE's
df_mse=pd.DataFrame({'LR':[mse1], 'DT':[mse2],
                      'RF':[mse3], 'kNN':[mse4],
                     'SVM-linear': [m_mse[0]],
                     'SVM-rbf': [m_mse[1]],
                     'SVM-poly': [m_mse[2]],
                     'SVM-sig': [m_mse[3]] 
                     })

print(df_mse)

# transpose the data
df_mse.T

    
# create a dataframe to store the Actual / Predicted values

df_vals = pd.DataFrame({'actual':testy,
                        'LR':p1,
                       'DT':p2,
                       'RF':p3,
                       'kNN':p4,
                     'SVM-linear': p5[0],
                     'SVM-rbf':  p5[1],
                     'SVM-poly': p5[2],
                     'SVM-sig': p5[3] } )

df_vals
    
# visualise the Actual vs Predicted Data
def showPlot(act,pred,model):
    ax1=sns.distplot(act,hist=False,color='r',label='Actual')
    sns.distplot(pred,hist=False,color='b',label='Predicted',ax=ax1)
    plt.title('Actual vs Predicted. Model = ' + model)
    

# function call to display the chart
showPlot(df_vals.actual,df_vals['SVM-rbf'],'rbf')  
    