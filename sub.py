import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import re
from scipy import sparse
from sklearn.linear_model import Ridge
import gc 

train = pd.read_csv('input\\train.tsv', sep = '\t', index_col=['train_id'])
test = pd.read_csv('input\\test.tsv', sep = '\t', index_col=['test_id'])
test_id = test.index
y = train['price'].apply(np.log1p)
nrow = train.shape[0]
data = pd.concat([train, test])

del train
del test
gc.collect()

categories = data['category_name'].str.split('/', n = 2, expand=True)
data['category1'], data['category2'], data['category3'] = \
categories[0], categories[1], categories[2]

del categories
gc.collect()

def high_low_price(groupby):
    groupby_median = data.groupby(groupby)['price'].median()
    high_price_99 = groupby_median.quantile(.99)
    low_price_01 = groupby_median.quantile(.01)
    high_price_cat = groupby_median[groupby_median > high_price_99].index.tolist()
    low_price_cat = groupby_median[groupby_median < low_price_01].index.tolist()
    return high_price_cat, low_price_cat

cat1_high, cat1_low = high_low_price('category1')
cat2_high, cat2_low = high_low_price('category2')
cat3_high, cat3_low = high_low_price('category3')
brand_high, brand_low = high_low_price('brand_name')

data.drop('price', axis = 1, inplace = True)

data['cat1_high'] = np.where(data['category1'].isin(cat1_high), 1, 0)
data['cat2_high'] = np.where(data['category2'].isin(cat2_high), 1, 0)
data['cat3_high'] = np.where(data['category3'].isin(cat3_high), 1, 0)
data['cat1_low'] = np.where(data['category1'].isin(cat1_low), 1, 0)
data['cat2_low'] = np.where(data['category2'].isin(cat2_low), 1, 0)
data['cat3_low'] = np.where(data['category3'].isin(cat3_low), 1, 0)
data['brand_high'] = np.where(data['brand_name'].isin(brand_high), 1, 0)
data['brand_low'] = np.where(data['brand_name'].isin(brand_low), 1, 0)
pop_brand = data['brand_name'].value_counts()[:50].index.tolist()
data['pop_brand'] = np.where(data['brand_name'].isin(pop_brand), 1, 0)


data.drop(['category_name'], axis=1, inplace=True)
data['item_description'].fillna('Unknown description', inplace=True)
data['brand_name'].fillna('Unknown brand', inplace=True)

#sig_brand = data['brand_name'].value_counts().index.tolist()[1:3001]
#data.loc[data['brand_name'].isin(sig_brand), 'brand_name'] = 'Unknown brand'

data['category1'].fillna('Unknown category1', inplace=True)
data['category2'].fillna('Unknown category2', inplace=True)
data['category3'].fillna('Unknown category3', inplace=True)

#sig_cat2 = data['category2'].value_counts().index.tolist()[:70]
#sig_cat3 = data['category3'].value_counts().index.tolist()[:400]
#data.loc[~data['category2'].isin(sig_brand), 'category2'] = 'Unknown category2'
#data.loc[~data['category3'].isin(sig_brand), 'category3'] = 'Unknown category3'


count_name = CountVectorizer(min_df = 10)
X_name = count_name.fit_transform(data['name'])

count_category1, count_category2, count_category3 = CountVectorizer(), CountVectorizer(), CountVectorizer()
X_cat1 = count_category1.fit_transform(data['category1'])
X_cat2 = count_category2.fit_transform(data['category2'])
X_cat3 = count_category3.fit_transform(data['category3'])

def preprocess(text):
    text = re.sub('(\d{2})\s?(gb?)', '\\1\\2', text, flags=re.IGNORECASE).upper()
    return text

count_des = TfidfVectorizer(max_features=40000, ngram_range=(1,3), stop_words='english', \
                           preprocessor = preprocess)
X_dec = count_des.fit_transform(data['item_description'])

brand_label = LabelBinarizer(sparse_output=True)
X_brand = brand_label.fit_transform(data['brand_name'])

other_col = ['cat1_high','cat2_high', 'cat3_high', 'cat1_low', 'cat2_low', 'cat3_low',
       'brand_high', 'brand_low', 'pop_brand']
X_other = sparse.csr_matrix(np.c_[pd.get_dummies(data[['item_condition_id', 'shipping']]), 
                                        data[other_col]])

X_data = sparse.hstack((X_name, X_cat1, X_cat2, 
                          X_cat3, X_dec, X_brand, X_other)).tocsr()


X = X_data[:nrow]
X_test = X_data[nrow:]

ridge = Ridge(solver='sag', random_state = 0, alpha = 1)
ridge.fit(X, y)
ridge_pred = ridge.predict(X_test)


sub = pd.DataFrame({'test_id': test_id, 'price':np.expm1(ridge_pred)})
sub.to_csv("Ridge Submission", index=False)







