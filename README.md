# MACHINE-LEARNING-USING-PYTHON-

hello there!!

I will use python for machine learning and also those who knows R can use it also, as python an R having same type of coding.

Now we will start from : 

# DATA PREPROCESSING
TOOLS:
# Importing the libraries

numpy// allow us to work with arrays.

matplotlib// alow us to import graphs and charts.

pandas// import datasets and create matrix of features and the dependent variable vector.

start with :
  import numpy as np   //shortcut for numpy is np
  
  import matplotlib.pyplot as plt // (.) will indicate here that matplot library and the modules we choose is by plot and
                                     shortcut as plt.
                                     
  import pandas as pd //shortcut pd
  
# Importing the dataset:
create a variable

  call pandas library then include .  
  dataset = pd.read_csv('Data.csv')
  now important principle is include features containing the information on which the dataset is dependent; generally you will find it in the last column
 
x = dataset.iloc[:, :-1].values
  
x is matrix of features
play with the indexes of all indexes except the last one
iloc//locate indexes
[:, :-1] for rows and the space with: is used for indicating the index 0 it will include but -1 is to exclude the last one.
.values will help you to get the dataset values

now, we will do same for the dependent variable vector (y)

y = dataset.iloc[:, -1].values

[:, -1] it will include only last column.
now you can print 
print(x)
print(y)

 you can import your dataset to get the print values
 
# oop : classes and objects
A class is the model of something we want to build. For example, if we make a house construction plan that gathers the instructions on how to build a house, then this construction plan is the class.

An object is an instance of the class. So if we take that same example of the house construction plan, then an object is simply a house. A house (the object) that was built by following the instructions of the construction plan (the class).
And therefore there can be many objects of the same class, because we can build many houses from the construction plan.

A method is a tool we can use on the object to complete a specific action. So in this same example, a tool can be to open the main door of the house if a guest is coming. A method can also be seen as a function that is applied onto the object, takes some inputs (that were defined in the class) and returns some output.

# Taking care of missing data
for eg: if there is missing value in a particular column and have a large number of dataset then we can simply delete it or if there are too many of missing details in the column then we have to replace it from the average value of the particular column.

we will use sklearn it is used in machine learning too much times

from sklearn.impute import SimpleImputer //object of SimpleImputer class is imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') //an empty value replace by mean
imputer.fit(X[:, 1:3])       //now we ave to connect them ,having methods  fit method will compute the values 
X[:, 1:3]=imputer.transform(X[:, 1:3]     //another method will now have to be used too get down the values it will also update the matrix so we replace it from the new value (average value) .

# Encode categorial data
# Encode independent variables
why to use this????
so, the answer is here
encode the country name with numbers strings into numbers and for eg:if the column is having the three entries then we will give number its not a corelation between the country and then we divide that column into three columns if there were 5 entries then 5 columns and allocate them their vector like binary numbers.
then if there is another column having the characters entries also replace them by the binary entries.
 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers[('encoder', OneHotEncoder(),[0] )],remainder='passthrough')

//create an object variable ct call class itself and enter the arguments then the arguments include transformers having elementsencoding then indexes then comes remainder that column which does not need any updation passthrough like age and salary

X = np.array(ct.fit_tranform(X))
//we do not have to use two steps for transform we will use only one step that will fit the
with three column of the country matrix hen we use numpy array to have the output of fit_transform
print(X)
//now, we have the three columns having the unique id vectors and converted into the numbers
//now,we convert the other column strings into the 0's and 1's

# Encode dependent variables

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
//we dont need numpy array here 
print(y)
//now it will convert into 0 and 1.

# Splitting the dataset into the training set and the set

We need to feature scaling before the split or after the split????

We have to feature scaling after the split. Two set first machine learning model where to train your dataset on exsisting set  and second to evaluate the performance of your model on new observation and deploy your dataset set on new different sets. feature scaling consist of scaling all ur features,t hey all take values in same scale to prevent one feature dominate other .

It's the simple reason test set on which u evaluate your dataset, feature scaling the technique comes on mean and standard deviation in the test set. would cause data / information leakage in the dataset.

from sklearn.model_selection import train_test_split //function -> (train_test_split)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=1)
//four seperate set for train set and test set help in prediction infuture ML models. now we will input matrix features(X) and dependent variable (y) train set 80% observations and test set 20% observations 
print(X_train)
//now 8 observations 8 customers randomly dumy variable,age,salary
print(X_test)
//2 obseravtions 2 customers randomly dumy variable age and salaries
print(y_train)
//8 purchased decisions correspond of X_train decisions
print(y_test)
//2 purchase decision in correspond to y_train

# Feature Scaling

All our features in same scale, some of the ml models thats the features do not dominate each other. In some of the dataset it is not all the time.
 
 Types of feature scalaing:

Standardisation 
consist of each value of the mean value and divide it from standard deviation of the feature
in -3 to -2
x(stand)=x-mean(x)/standard deviation(x)

Normalisation
consist of each value of the feature and substract it from minimumx feature divide it from the maximum value of feature and substracting from minimum value of feature
in 0 to 1
What should we go for????
 //standardisation work all the time.
 //normalisation work normal distribution in your features ,more recommended for the particular type.
 
 from sklearn.preprocessing import StandardScaler// preprocesing module
 sc = StandardScaler()
 //do we have to the feature scaling on dumy variable?? no, goal is to have the value features into same scale -3 to +3 since dumy varibles are alreafy having values-1 to -2
you will lose the information,get nonsense value totally lose interpretation .leave dumy variable as its is.
matrices of features
X_train[:, 3:]=sc.fit_transform(X_train[:, 3:])//only two columns age and salary age column has index 3 and 4 for salary 3: from index 3 to uptil all the columns. fit get u the mean and standard deviation and tranform will apply the formula of standardisation and get the value.
now,
X_test[:, 3:]=sc.tranform(X_test[:, 3:])//new data in production we will only apply transform method x_test predict the model with the particular scalar 

print(X_train)  
print(X_test)

# REGRESSION
Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

In this part, you will understand and learn how to implement the following Machine Learning Regression models:

Simple Linear Regression
Multiple Linear Regression
Polynomial Regression
Support Vector for Regression (SVR)
Decision Tree Classification
Random Forest Classification

# Simple linear regression
*y = b0+b1*x1
dependent variable DV (Y)
x1(independent variable) IV
b1 (cofficient)
*graph
x axis having experience and y axis with salary
*salary = b0+b1*experience
point where it touches the y axis  eg:30k is b0
and b1 is slope of the line for eg: how his salary project to y-axis if the b1 is less slope is experience will be yield less in the salary.

 now draw the verticle line to the line so the dots will actual observation
 for eg: 5 years of experience having salary $100k the verticle line touches the slant line then at that point is the actual modelled value it will prediction of ernings 
 differnce between actual and modelled 
 sum(y-y1)^2 ---->min
 //records in the temporary file
 that is the best fitting line
 
SIMPLE LINEAR REGRSSION
calculating years of experience and salary trained the corelation between the years of experience and corresponding salary.
# Importing the libraries
first step is datapreprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset=pd.read.csv('Salary_Data.csv')
X=dataset.iloc[:,  :-1].values
y=dataset.iloc[:,  -1].values
# Splittting the dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_slit(X,y,test_size=0.2,random_state=0)
# Training the simple linaer regression model on the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()//regressor is the object of class LinearRegression,usually there is parameters but now we dont need it we get a model,now we have to build,train and connect in next step.
regressor.fit(X_train,y_train)//having methods like train and predicting futuretrain the smple linear regression fix method has a certain format pattern x having features of independent variables of training set and y having dependent variable vector features.
# Predicting the test set results 
method to use is to use predict method
predict the observation in the dataset we split it into 80 % train set and 20% data set 
last six salaries are the true and predicted salaries 6 and then relate them
y_pred=regressor.predict(X_test)//put them in a vector having predicting salaries
# Visualising the training set results
plt.scatter(X_trainy_train, color='red')// scatter allow us put the red points means the real salaries co ordinate in the graph.
plt.plot(X_train,regressor.predict(X.train),color = 'blue')//plot the regression line is the line of prediction close as possible to the real salaries follow  staright line as it is linear,visualising the training set in x coordinate and y axis co ordinate predicting salaries.
//now plot a nice graph
plt.title('Salary vs Experience ( Training set )')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
//this will show the graphic
# Visualising the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
//this will show the graphic

# Multiple Linear Regression
EG:
TO ANALYSE THE 50 COMPANIES WHICH TYPE OF COMPANY IS MORE INTERESTED IN INVESTMENT AND CREATE A MODEL.
HAVING R&D SPEND,ADMINISTRATION,MARKETING,STATE AND PROFIT WITH THESE COLUMNS.
WHICH COMPANY IS HAVING MORE PROFIT ,SO WHERE IN WHICH COMPANY TO INVEST?

*y = b0 + b1 * x1 + b2 *x2+.....+bn*xn
salary how much years of experience,how much courses u have done,how much money u have made for the company, how much student lecture attended and how much time he slept, how much time he studied.

# Assumptions of Linear Regression :
linearity,
homoscedasticity,
multrivarite normality,
independence of errors,
lack of multicollinearity

dumy variables:
any correlation between r&d,marketing or profit,administration to predict profit.
profit > dependent variable
and all the other are independent variables.

*y = b0+b1*x1+b2*x2+b3*x3+b4*D1

b4*D1 >>>>>> for state column we dont have any numericl and we cannt enter the string so we categorial variable so we cannt add it.
create dumy variable, how many categries are there there is for eg: 2 state one is A and second is B number it 1 and 0 and build 2 columns for rows whch says which number them 0 and 1.
And these two columns as dumy variables.>>b4 * d1 its for A state. why we should not use all the dumy variable only one is enough as where it is 1 then this company is in A state and if 1 then it is in company B.
b3*x3 >>>>> marketing spend
b2*x2 >>>>>> admin variable 
b1 * x1 >>>>>>> r&d spend
b0 >>>>>>> constant

And why it is bad idea to include both dumy variable???

duplicating a variable,beacause
*d2=1-d1

multicollinearity the model cannt distinguish the both effect,the real problem is u cannt have constant and both dumy variable at a time it is not possible .
if u have 99 different state then include 98 ,if u have 6 include only 5.

*WHAT IS P-VALUE????

H0 IS THE AGAINST EVIDENCE
H1 IS THE EVIDENCE WE TRY TO PROVE
significance level is p value is low then null hypothesis then it will reject.
p large we do not reject the null hypothesis.
p value tells the evidance .

*Building a model:

Before we have only one dependent and one independent variable. 
now we have so many of columns are potential predictors for a dependent variables.there are so many and we have to decide the column which to keep and which to throw.

garbage in and garbage out wont be reliable and became garbage model ,and u have to describe them all.
we have to keep the important one.

*5 methods of building model:

all in
backward elimination       //step wise regression
forward selection.         //step wise regression
bidirectional elimination  //step wise regression
score comparision

*all-in

prior knowledge,you have to to use specific variables,preparing for backward elimination.

*backward elimination

step1: select a significance model to stay in model(eg : sl=0.05).
step2: fit the full model with all possible predicators.
step3: consider the predictor with the highest p-value it p>sl ,go to next step otherwise go to fin.
step4: remove the predictor.
step5: fit model ithout this variable.

*forward selection

step1: select a significance model to stay in model(eg : sl=0.05).
step2: fit all simple regression models y~xn .select the one with the lowest p-value.
step3: keep this varable and fit all possible models with one extr predictor added to the one(S) you already have.
step4: consider te predictor with the lowest p-value. If, p<sl ,go to step3 ,otherwise go to fin.

you dont keep the current model keep previous model.

*bidirectional elimination

step1: select a significance model to stay in model(eg : SLENTER=0.05, SLSTAY=0.05).
step2: perform the next step of the forwaed selection(new variables must have: P<SLENTER to enter).
step3: perform all the step of backward elimination(old variable must have P<SLSTAY to stay).
step4: no new variable. can enter and no old variable can exit.

 *all possible models
 
 step1: slect a criterion of goodness of fit.
 step2: construct all possible regrssion models:(2^n)-1 total combinations.
 step3: select the one with best criterion.
 
 example:
 10 columns means 1023 models.
 throwing everything in no of models incrase exponentially,its very resource consuming .
 so we have total 5 models for building a model.
 we usebackward elimination as it is efficient enough.
 
In which startup to invest having an dataset for eg : of 50companies as we have discussed above.

we check all the columns that they are having numerical values and having no missing data.
check independent and dependent variable.
and if there is any column having not numerical variable having state names, x y and z.
we have to encode them into categorial variable.
look your dataset if it is no long and if it is long then check it by data preprocessing tool kit.

start with multiple linear regression
 
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('50_Startups.csv') 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorial data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],  remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X) //encoding of dataset 3 new columns at the begining
//we dont have to aplly featuring because in multiple linear regression there is cofficients in each term that will scale them so there is no need of feature scaling

# Splitting the dataset into the trining set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the multiple linear regression model on the training set

Do we have to avoid the dumy variable trap??
no,the datset we build and train will automatically avoid the trap.

Do we have to work on features like backward elimination having high pvalue??
no the class we have to build will automatically identify the best features and  predict the high pvalue with highest accuracy.

sklearn library will help you a lot.
 
from sklearn.linear_moel import LinearRegression
regressor = LinearRegression() //regressor is the object of LinearRegression class
regrssor.fit(X_train, y_train) //.fit is the method

# Predicting the test set result 
we cannt plot the graph heare because heare are 5 columns and we can not display it.

y_pred= regressor.predict(X_test)//X test .... not including profit column 
np.set_printoptions(precision=2)//display any numerical value after decimal only 2 values
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


//concatenate 2 vectors tuple of arrays or vectors u want to concatenate having same shape,meaning same number of profit becuase we want print them vertically not horizontally use reshape enter no of columns use length function y pred and ,1 means one column so this is for the fist element now we want to concate the next vector with real profits using y_test

0 means horizontal conctenation and 1 means vertical.

after running the program,,
left side we have the vectors of predicted profit nd at the right side we have real profit and now compare them some are very close predictions and some are okay.

# Polynomial Regression

*y=b0+b1x1+b2(x1)^2+....+bn(x1)^n

so the square will give the parabolic turn in the graph.
sometimes polynomail regression is really helpful for the pandemics,endemics,etc.

it is also called the polynomial linear regression.
by seeing the cofficients by this it is linear 

datset consisting of 3 columns first position then level and salary
build the model using polynomial regresion and by collecting data such as salary havng position from business analyst to CEO.

see through linkedln from how many years and we will predict the salary.

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

we have very few dataset so we do not split the data as it is so small so we do not apply it here.

# Importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
As linear regression compare it with as polynomial model is much more udapted to this dataset.

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
//we have build simple linear regression.
now we will build ultiple linear regression with the powers and also integrate the powers in linear regression.

# Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)//degree =4 is used here for
X_poly = poly_reg.fit_transform(X) //fit_tranform will help matrix feature one into converting the new matrix features x1,x1^2,x1^3,x1^4
lin_reg_2 = LinearRegression()//now we have to add b1,b1,...cofficients 
lin_reg_2.fit(X_poly,y)//now u have build the polynomial linear regression

# Visualising the Linear Regression results

plt.scatter(X,y,color= 'red')//displat 2d plot conataining the real results..here for eg : X position level y is predicted salaries.
plt.plot(X, lin_reg.predict(X),color = 'blue')//actully going to plot the blue line 
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')//x is simple linearregressor so we use poly_reg
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
//and now we got the regression curve comes near the real salaries as compare from the before one.
if n=2 and if we increase the power it will be more accurate for eg with degree 4 it will be so accurate. But it is not smooth curve .

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1)//instead of taking integers 1,2,3.....we will incrase the density .1,.2,.3,......
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
we will predict the salary first with linear regression and second with multiplt regression.

lin_reg.predict([[6.5]])//predict the salary of position 6.5 we have to input the the number then input it with array conatining [[]]it means 2d aarray 1 is for row and second for column
. Before execute we check the salary and from this the prediction is over the real, so it is wrong.

# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

//and now we got the predicted salary from polynomial regression which is super close to the the real dataset salary.

# SVR support_vector_regression

invented by vladmir vapnik 
learn about support vector machine and support vector regression

we will plot a graph of having the simple linear regression and plot accordingly ordinary least squares sum(y-y1)->min
now having another graph with same dots having regression line in between and having a tube with width ephsiln on both side think it is it of as error line 
and the area falls between wll hving a buffer of error and we dont care about error for these co ordinates but the coordinates that fall outside the ephsiln insensitive tubes , these and we care about these buffers so the co ordinates or ponts which are below the tube denoted as c*  and the points above the tube denoted as c these variable are slack variables and we care about them as they have error so we use formula:

and all the points are the 2d vectors the outside ones are supporting vectors.












