import ryan_tools as rt
import seaborn as sea
import pandas as pd
import sklearn as sk
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

classifier = tree.DecisionTreeRegressor()
lb = LabelEncoder()

data = pd.read_csv('amazon.csv')

data['type2'] = lb.fit_transform(data['type'])


data['type3'] = data['type'] + (data['type2'].apply(str))
print(data['type3'].unique())
data['fulfillment_map'] = 0
data['fulfillment'] = data['fulfillment'].fillna('None')
data.loc[data['fulfillment'].str.contains('Seller', case = False ), 'fulfillment_map'] = 1
data.loc[data['fulfillment'].str.contains('Amazon', case = False ), 'fulfillment_map'] = 2


classifier.fit( data[['type2','product sales', 'shipping credits', 'fulfillment_map']] , data['selling fees'] )

def predict():
    order_type = input('Order type?\n')
    sales = input('Product Total?\n')
    shipping = input('Shipping Amount?\n')
    fulfillment = input('fulfillment? 1 = FBSELLER, 2 = FBA , 0 = None\n')
    find_me = [order_type , sales, shipping,fulfillment]
    array_container = []
    array_container.append(find_me)
   
    print(classifier.predict( array_container ))
    return array_container
