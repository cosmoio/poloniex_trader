import pandas as pd
import numpy as np
import json

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


from util.message import print_message
from util.message import print_summary
from util.message import print_logo

def build_prediction_forest(df):
    training_size = 0.3         # Training size between 0 and 1

    print_message("Reading dataset from "+filename,"information")
    with open(filename) as data_file:
        data = json.load(data_file)


    print_message("Creating dataframe","information")
    df = pd.DataFrame(data)

    
    #do_ml(df)


    print_message("Creating new features from indicators","information")

    #dff = compute_crtdr(dff)
    #dff = compute_pivot(dff)
    #dff = compute_r1(dff)
    #dff = compute_r2(dff)
    #dff = compute_r3(dff)
    #dff = compute_s1(dff)
    #dff = compute_s2(dff)
    #dff = compute_s3(dff)
    #dff = compute_rsi(dff)
    
    # Special feature for the detection later on
    #dff = compute_return(dff)    

    print_message("Defining training set","infomration")

    num_training_samples = len(df) * training_size
    if training_size > len(df) or training_size < 0:
        print_message("Training size has to be a value between 0 and 1","error")
        sys.exit()
        
    print_message("Training sample size: "+str((num_training_samples/len(df))*100)+" of dataset","success")
    dff["is_train"] = df.index < (len(df) * training_size)


    print_message("Creating binning labels for performance","information")

    dff['LABEL'] = pd.cut(dff['RETURN'], 7, labels=["dysmal","terrible","bad","meh","good","great","fantastic"])
    #print(dff[dff.LABEL == "dysmal"])
    #print(dff[dff.LABEL == "terrible"])
    #print(dff[dff.LABEL == "bad"])
    #print(dff[dff.LABEL == "okay"])
    #print(dff[dff.LABEL == "good"])
    #print(dff[dff.LABEL == "great"])
    #print(dff[dff.LABEL == "fantastic"])
    #[(-0.0371, -0.0249] <       dysmal
    # (-0.0249, -0.0128] <       terrible
    # (-0.0128, -0.000715] <     bad
    # (-0.000715, 0.0114]  <     okay
    # (0.0114, 0.0235] <         good
    # (0.0235, 0.0356] <         great
    # (0.0356, 0.0477]],         fantastic
    #    print(dff[dff.LABEL == "fantastic"])

    counts = pd.value_counts(dff['LABEL'] )
    #print(counts)

    print_message("Defining features", "success")
    
    dff = dff.drop('date', 1)   # removing date column
    
    features = dff.columns[:17]

    print_message("Creating training and test instances","information")
    train, test = dff[dff['is_train']==True], dff[dff['is_train']==False]
    
    print_message("Number of observations in the training data:" + str(len(train)),"success")
    print_message("Number of observations in the test data:" + str(len(test)), "success")


    print_message("Factorize goal feature","information")
    # train['RETURN'] contains the actual returns. Before we can use it,
    # we need to convert each return a digit. So, in this case there
    # are three species, which have been coded as 0, 1, or 2.
    #    dff = compute_label(dff)
    #categories = pd.cut(dff['RETURN'], [-10,-5,0,5,10], retbins=True)#labels=["bad","okay","good"])
    #print(categories)
    
    y = pd.factorize(train['LABEL'])[0]
    
    print_message("Creating RandomForest Classifier instance: jobs=2, trees=default","information")
    clf = RandomForestClassifier(n_jobs=2)

    print(y) 
    for x in list(zip(train["RETURN"], train["LABEL"])):
        print(str(x))
            
    print_message("Fit model to training data","information")
    clf.fit(train[features], y)


    print("Test predictions","information")
    clf.predict(test[features])

    #print(clf.predict_proba(test[features])[0:10])
    # Create actual english names for the plants for each predicted plant class
    labels = ["dysmal","terrible","bad","meh","good","great","fantastic"]
    
    labels_arr = array(labels)
    print(labels)
    #print(clf.predict(test[features])[0:300])
    
    #print(pd.DataFrame(clf.predict_proba(test[features]), columns=clf.classes_))
    ##iris = load_iris()
    #print(type(iris.target_names))
    #print(type(labels))
    
    #preds = clf.predict(test[features])
    #wfor idx in preds
    preds = labels_arr[clf.predict(test[features])]
    #print(preds[0:2000])
    #y = pd.factorize(test['LABEL'])[0]
    #print (accuracy_score(preds, y))*100 #prediction accuracy

    print(pd.crosstab(test["LABEL"], preds, rownames=['Actual Label'], colnames=['Predicted Label']))
    # View the PREDICTED species for the first five observations
    #preds[0:5]

    print_message("Feature importance","information")
    
    # View a list of the features and their importance scores
    for x in list(zip(train[features], clf.feature_importances_)):
        print_message(str(x),"information")

    from sklearn import tree
    i_tree = 0
    for tree_in_forest in clf.estimators_:
        with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
        i_tree = i_tree + 1


def compute_crtdr(df):
    print_message("Creating \"CRTDR\" indicator","information")
    
    df["CRTDR"] = ((df.close-df.low)/(df.high-df.low)) > .5
    
    print_message("CRTDR: Done","success")
    return df
    
def compute_return(df):
    print_message("Creating \"RETURN\" indicator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    close_column = df["close"] 
    
    df["RETURN"] = 0
    
    for index, row in df.iloc[1:].iterrows():
        df.ix[index,"RETURN"] = (df.ix[index]["close"] - close_column.ix[index-1]) / close_column.ix[index-1]
    
  #  df["close"]i-1 - df["close"] / df["close"]i-1
    
    print_message("\"RETURN\": Done","success")
    return df

def compute_pivot(df):
    print_message("Creating \"PIVOT\" indicator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    df["PIVOT"] = df["high"] + df["low"] + df["close"] / 3    
    
  #  df["close"]i-1 - df["close"] / df["close"]i-1
        
    print_message("\"PIVOT\": Done","success")
    return df

def compute_r1(df):
    print_message("Creating \"Resistance R1\" indicator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    df["R1"] = 2* df["PIVOT"] - df["high"]    
    
  #  df["close"]i-1 - df["close"] / df["close"]i-1
        
    print_message("\"R1\": Done","success")
    return df

def compute_r2(df):
    print_message("Creating \"Resistance R2\" indicator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    df["R2"] = df["PIVOT"] + df["high"] - df["low"]    
    
  #  df["close"]i-1 - df["close"] / df["close"]i-1
        
    print_message("\"R2\": Done","success")
    return df

def compute_MACD(df, n_short,n_long):
    print_message("Creating \"MACD\"  Indicator","information")

    df['EMA_'+str(n_short)] = pd.ewma(df["close"], span=n_short)
    df['EMA_'+str(n_long)] = pd.ewma(df["close"], span=n_long)

    df['MACD'] = (df['EMA_'+str(n_short)] - df['EMA_'+str(n_long)])

    df['MACD_sign'] = pd.ewma(df['MACD'], span = 9, min_periods = 8)
    df['MACD_diff'] = df['MACD'] - df['MACD_sign']
    
    
    print_message("Done: \"MACD\"  Indicator","success")
    return df



#Momentum  
def compute_MOM(df, n):  
    print_message("Creating \"MOM\"  Indicator","information")
    
    M = pd.Series(df['close'].diff(n), name = 'MOMENTUM_' + str(n))  
    df = df.join(M)  

    print_message("Done: \"MOM\"  Indicator","success")
    return df

def compute_r3(df):
    print_message("Creating \"Resistance R3\" indicator","information")
   
    df["R3"] = 2* (df["PIVOT"] - df["low"]) + df["high"]    
    print_message("\"R3\": Done","success")
    return df

def compute_s1(df):
    print_message("Creating \"Support S1\" indicator","information")
 
    df["S1"] = 2* df["PIVOT"] - df["high"]   
    print_message("\"S2\": Done","success")
    return df


def compute_s2(df):
    print_message("Creating \"Support S2\" indicator","information")

    df["S2"] = df["PIVOT"] - (df["high"] - df["low"])   
    print_message("\"S2\": Done","success")
    return df

def compute_s3(df):
    print_message("Creating \"Support S3\" indicator","information")
   
    df["S3"] = df["low"] - 2 * (df["high"] - df["PIVOT"])   
    print_message("\"S3\": Done","success")
    return df

def compute_rsi(df):
    print_message("Creating \"RSI\" indicator","information")
   
    close_column = df["close"] 
    
    df["RSI"] = 0
    
    sum_neg = 0
    sum_pos = 0
    n = 14
    rows = len(df)
    print_message("Computing RSI for each row","log")
    x = 1
    while x < rows: #for x, row in df.iloc[n:].iterrows():
        #print_message("Computing row: "+str(x)+"/"+str(rows),"log")
        if x+n < rows-1:
            for y in range (x,x+n): 
                if df.ix[y]["close"] < close_column.ix[y-1]:
                    sum_neg += (close_column.ix[y-1] - df.ix[y]["close"])
                elif df.ix[y]["close"] > close_column.ix[y-1]:
                    sum_pos += (df.ix[y]["close"] - close_column.ix[y-1])
                

            sum_pos/=n
            sum_neg/=n
            
            if sum_neg == 0:
                break;
            rs = sum_pos/sum_neg
            rsi = 100 - 100/(1+rs)
            df.ix[x,"RSI"] = rsi
            
            sum_neg = 0
            sum_pos = 0
            #print(str(y)+" "+str(x)+" "+str(df.ix[x,"RSI"]))
        x+=1
    print_message("\"RSI\": Done","success")
    print(df["RSI"].head())
    print(df["RSI"].tail())
    
    return df;

def compute_label(df):
    bins = [-4.0,0,4.0,8.0]
    group_names = ["Minus","Okay","Great"]
#    group_names = ["Terrible","Bad","Minus","Even","Okay", "Good","Great"]

    categories = pd.cut(df['RETURN'], 3, labels=group_names)
    #df['LABEL'] = pd.cut(df['RETURN'], 3, labels=group_names)
    print(categories)
    #pd.value_counts(df['LABEL'])
    sys.exit()
    return df
