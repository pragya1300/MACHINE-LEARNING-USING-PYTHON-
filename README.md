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
 
# x = dataset.iloc[:, :-1].values
  
x is matrix of features
play with the indexes of all indexes except the last one
iloc//locate indexes
[:, :-1] for rows and the space with: is used for indicating the index 0 it will include but -1 is to exclude the last one.
.values will help you to get the dataset values

now, we will do same for the dependent variable vector (y)

# y = dataset.iloc[:, -1].values

[:, -1] it will include only last column.
now you can print 
# print(x)
# print(y)

 you can import your dataset to get the print values
 

