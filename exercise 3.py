import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def feature_plot (filename, df,feature):

    df.hist(column=feature);
    plt.show()
    fn=filename+'.png'
    plt.savefig(fn)
    plt.clf()

    print('feature ', feature, ' : mean=', df.iloc[:,feature-1].mean(), ' std=', df.iloc[:,feature-1].std())
    if feature == 3:
        print(' class distribution (ie for column ', df.groupby(df.iloc[:,4]).size())


#read input file
fin = open("iris.data", "rt")
#read file contents to string
data = fin.read()
#replace all occurrences of the required string
#STEP 1:
data = data.replace('Iris-setosa', '1')
data = data.replace('Iris-versicolor', '2')
data = data.replace('Iris-virginica', '3')
#close the input file
fin.close()
#open the input file in write mode
fin = open("iris.data", "wt")
#overrite the input file with the resulting data
fin.write(data)
#close the file
fin.close()

#STEP 2 and 3 and 4:
df=pd.read_table("iris.data", header=None, sep=',')

for i in range(0,4):
    feature_plot ('feature'+str(i), df,i)

#STEP 5
# create training and testing vars

train_random, test_random = train_test_split(df, train_size=0.6)

train_not_random, test_not_random = train_test_split(df, train_size=0.6, shuffle=False)

#STEP 6:
for i in range(0,4):
    feature_plot ('train_random_feature'+str(i), train_random,i)

for i in range(0,4):
    feature_plot ('test_random_feature'+str(i), test_random,i)

for i in range(0,4):
    feature_plot ('train_not_random_feature'+str(i), train_not_random,i)

for i in range(0,4):
    feature_plot ('test_not_random_feature'+str(i), test_not_random,i)

#STEP 7:
feature_selected = int(input("Select feature (enter 0, 1, 2, or 3) "))

train_random_selected=train_random.loc[:,[feature_selected,4]]
test_random_selected=test_random.loc[:,[feature_selected,4]]
print (train_random_selected)
print (test_random_selected)

#STEP 8:

train_random_class_1_or_2=train_random.loc[train_random[4].isin([1,2])]
test_random_class_1_or_2=test_random.loc[test_random[4].isin([1,2])]
np.savetxt(r'binary_iristrain.txt', train_random_class_1_or_2.values, fmt=['%1.1f','%1.1f','%1.1f','%1.1f','%d'])
np.savetxt(r'binary_iristest.txt', test_random_class_1_or_2.values, fmt=['%1.1f','%1.1f','%1.1f','%1.1f','%d'])

for i in range(0,4):
    feature_plot ('binary_iristrain_feature'+str(i), train_random_class_1_or_2,i)
for i in range(0,4):
    feature_plot ('binary_iristest_feature'+str(i), test_random_class_1_or_2,i)

#STEP 9:
scaler=MinMaxScaler()
scaled_values=scaler.fit_transform(train_random_class_1_or_2.loc[:,0:3])
train_random_class_1_or_2_scaled=train_random_class_1_or_2
train_random_class_1_or_2_scaled.loc[:,0:3]=scaled_values
scaled_values=scaler.fit_transform(test_random_class_1_or_2.loc[:,0:3])
test_random_class_1_or_2_scaled=test_random_class_1_or_2
test_random_class_1_or_2_scaled.loc[:,0:3]=scaled_values
np.savetxt(r'binary_iristrain_scaled.txt', train_random_class_1_or_2_scaled.values, fmt=['%1.4f','%1.4f','%1.4f','%1.4f','%d'])
np.savetxt(r'binary_iristest_scaled.txt', test_random_class_1_or_2_scaled.values, fmt=['%1.4f','%1.4f','%1.4f','%1.4f','%d'])
