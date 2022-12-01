from sklearn.linear_model import LinearRegression
import pandas as pd
import scipy.spatial as sci
import timeit
import math
import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# @authors: Syed Muhammad Mustafa and Yousaf Khan

def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")

    demonstrateHelpers(trainDF)
    
    print(corrTest(trainDF))

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    
    doExperiment(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    

    
    
    

    
# ===============================================================================
def readData(numRows = None):
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    
    outputCol = ['SalePrice']
    
    return trainDF, testDF, outputCol
    
def corrTest(df):
    
    colNames = ['MSSubClass', 'LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '2ndFlrSF', '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'YrSold']
    salePrice = df.iloc[:,-1]
    #data, test, output = readData()
    
    corrDF = df.loc[:, colNames]
    
    corrFigures = corrDF.apply(lambda col: col.corr(salePrice) ,axis =0)
    
    return corrFigures
    


'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)
    
    '''
    We added GradientBoostingRegressor algorithm within the doExperiment Function
    '''
    
    alg1 = GradientBoostingRegressor()
    alg.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg1, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    

    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    trainInput = trainDF.iloc[:, :80]
    testInput = testDF.iloc[:, :]
    
   
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    
    
    
    'start preprocessing'
    
    '''
    We ran correlation test with all existing numerical columns and dropped all columns that had correlations between - 0.2
    and +0.3. This is because, we believe that such small correlation is not a good predictor for sales price.
    '''
    
    trainInput = trainInput.drop(['LotArea','OverallCond','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','EnclosedPorch','YrSold'], axis=1)
    testInput = testInput.drop(['LotArea','OverallCond','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','EnclosedPorch','YrSold'], axis=1)
    
    '''
    After inspecting the dataset we noticed that PoolQC, Fence, Alley, MiscVal attributes were almost entirely 'NA'. Hence, it does not
    provide anything valuable for our analysis. Therefore, we dropped it.
    We also realized that RoofMatl attribute had mostly 'CompShg' values. Following the same logic, it is not useful
    for our comparison. We dropped it.
    '''
    trainInput = trainInput.drop(['PoolQC','Fence','MiscVal','RoofMatl', 'Alley'], axis =1)
    testInput = testInput.drop(['PoolQC','Fence','MiscVal','RoofMatl','Alley'], axis =1)    
    
    
    '''
    Columns EnclosedPorch, 3SsnPorch, ScreenPorch, and PoolArea all have values that are predominately equal 
    to zero. This does not provide a good indication to the algorithm in making predictions. We had previously
    dropped EnclosedPorch as part of our correlation test. We will procede to drop the rest.
    
    '''
    trainInput = trainInput.drop(['3SsnPorch','ScreenPorch','PoolArea'], axis =1)
    testInput = testInput.drop(['3SsnPorch','ScreenPorch','PoolArea'], axis =1)    
    
    '''
    Standardized 
    '''
    
    standardizeCols = ['OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','2ndFlrSF','1stFlrSF','GrLivArea','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','LotFrontage']
    standardize(trainInput, standardizeCols )
    standardize(testInput, standardizeCols)
    
    '''
    Due to the ongoing nature of this checkpoint, we excluded certain attributes that have missing values. Additionally, we are using attributes that are numerical.
    
    '''
    
    predictors = ['OverallQual', 'YearBuilt','YearRemodAdd','2ndFlrSF','1stFlrSF','GrLivArea','Fireplaces','WoodDeckSF','OpenPorchSF']
    
    
    #predictors = ['MSSubClass','MSZoning','LotFrontage','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','YearBuilt','YearRemodAdd','RoofStyle','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','MiscFeature','MoSold','SaleType','SaleCondition']
    
    print(trainInput.loc[:,'OverallQual'])
    print(trainInput.columns)
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
    
# ===============================================================================
def standardize(inputDF, cols):
    inputDF.loc[:, cols] = (inputDF.loc[:, cols] - inputDF.loc[:, cols].mean()) /inputDF.loc[:, cols].std()
    return inputDF.loc[:, cols]
    



'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

