import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']

train = pd.read_csv('adult.data', sep=",\s", header=None, names = column_names, engine = 'python')
test = pd.read_csv('adult.test', sep=",\s", header=None, names = column_names, engine = 'python')
test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')


adult = pd.concat([test,train])
adult.reset_index(inplace = True, drop = True)


# print(adult.values.astype(str))
for col in set(adult.columns) - set(adult.describe().columns):
    adult[col] = adult[col].astype('category')



adult = adult[(adult.values.astype(str) == '?').sum(axis = 1) == 0]


cat_columns = adult.select_dtypes(['category']).columns
adult[cat_columns] = adult[cat_columns].apply(lambda x: x.cat.codes)


adult_data = adult.drop(columns = ['income'])
adult_label = adult.income


train_data, test_data, train_label, test_label = train_test_split(adult_data, adult_label, test_size  = 0.25)
# np.save("preprocessed_data/train_data.npy", train_data.values)
# np.save("preprocessed_data/test_data.npy", test_data.values)
# np.save("preprocessed_data/train_label.npy", train_label.values)
# np.save("preprocessed_data/test_label.npy", test_label.values)

print(test_data.shape)