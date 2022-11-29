#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
pd.set_option('display.max_columns', 100)
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
import json
import requests
import folium
import opendatasets as od
from datetime import datetime
import calendar
from pandas.api.types import CategoricalDtype
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
from collections import Counter
from PIL import Image


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# DataPreparation
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib


# In[4]:


# Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb


# In[5]:


#data reading
import time
start = time.time()
customer_data = pd.read_csv("olist_customers_dataset.csv")
location_data = pd.read_csv("olist_geolocation_dataset.csv")
item_data = pd.read_csv("olist_order_items_dataset.csv")
payment_data = pd.read_csv("olist_order_payments_dataset.csv")
review_data = pd.read_csv("olist_order_reviews_dataset.csv")
orders_data = pd.read_csv("olist_orders_dataset.csv")
products_data = pd.read_csv("olist_products_dataset.csv")
sellers_data = pd.read_csv("olist_sellers_dataset.csv")
category_data = pd.read_csv("product_category_name_translation.csv")
end = time.time()
print("reading time: ",(end-start),"sec")


# In[6]:


#checking number of columns , column_names and no_of_rows
datasets = [customer_data, location_data, item_data,payment_data,review_data, orders_data,products_data,sellers_data,category_data]
titles = ["customers","geolocations","items", "payments","reviews", "orders", "products","sellers","category_translation"]
df1 = pd.DataFrame({},)
df1['dataset']= titles
df1['no_of_columns']= [len(df.columns) for df in datasets ]
df1['columns_name']= [', '.join(list(df.columns)) for df in datasets]
df1['no_of_rows'] = [len(df) for df in datasets]
df1.style.background_gradient(cmap='PuBuGn')
#print(info_df)|


# 
# Observation(s):
# Dataset with maximum number of columns is products.
# Dataset with maximum number of rows is geolocations

# In[7]:


#checking dtypes

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df2 = pd.DataFrame({},)
df2['dataset']= titles

df2['numeric_features'] = [len((df.select_dtypes(include=numerics)).columns) for df in datasets]
df2['num_features_name'] = [', '.join(list((df.select_dtypes(include=numerics)).columns)) for df in datasets]
df2['object_features'] = [len((df.select_dtypes(include='object')).columns) for df in datasets]
df2['objt_features_name'] = [', '.join(list((df.select_dtypes(include='object')).columns)) for df in datasets]
df2['bool_features'] = [len((df.select_dtypes(include='bool')).columns) for df in datasets]
df2.style.background_gradient(cmap='Purples')
#print(new_df)


# Observation(s):
# products dataset has maximum number of numerical features(i.e dtype :'int16', 'int32', 'int64', 'float16', 'float32', 'float64').
# orders dataset has maximum number of features of object dtype
# We can also observe that all the timestamps are in object datatypes.So, we have to convert it into datetime type to do analysis on these features.

# In[8]:


#checking no of null values

df3 = pd.DataFrame({},)

df3['dataset']= titles

#creating column of name of columns in the dataset 
df3['cols'] = [', '.join([col for col, null in df.isnull().sum().items() ]) for df in datasets]

#creating total number of columns in the dataset 
df3['cols_no']= [df.shape[1] for df in datasets]

#counting total null values
df3['null_no']= [df.isnull().sum().sum() for df in datasets]

#creating total number of columns in the dataset with null-values 
df3['null_cols_no']= [len([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]

#creating column of name of columns in the dataset with null-values 
df3['null_cols'] = [', '.join([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]


df3.style.background_gradient(cmap='Purples')


# Observation(s):
# The maximum number of null-values are present in reviews dataset and the name of the columns with the null-values are review_comment_title and review_comment_message.
# products dataset contains least number of null- values but most of its columns has null-values.
# we have to deal with these null-values in future.

# In[9]:


#merging all data
rev_new = review_data.drop(['review_comment_title','review_creation_date','review_id','review_answer_timestamp'],axis=1)
df = pd.merge(orders_data,payment_data, on="order_id")
df = df.merge(customer_data, on="customer_id")
df = df.merge(item_data, on="order_id")
df = df.merge(products_data, on="product_id")
df = df.merge(category_data, on="product_category_name")
df = df.merge(rev_new, on="order_id")
df.head()


# In[10]:


print("Number of rows after merging:",len(df))
print("Number of columns after merging:",len(df.columns))


# In[11]:


#Handling missing values
df.isnull().sum()


# Handling Missing values in Timestamps
# 
# The order of different types of timestamps are shown below:
# order_purchase_timestamp-->order_approved_at--> order_delivered_carrier_date-->order_delivered_customer_date-->order_estimated_delivery_dat
# e     
# Timestamps containg missing values are order_approved_at, order_delivered_carrier_date, order_delivered_customer_date.
# 
# null-values in order_approved_at can be replaced by order_purchase_timestamp and null-values in order_delivered_customer_date can be replaced by order_estimated_delivery_date
# 
# we can drop the column order_delivered_carrier_date.
# 

# In[12]:


#Handling missing values
index = (df[df['order_delivered_customer_date'].isnull() == True].index.values)

df["order_approved_at"].fillna(df["order_purchase_timestamp"], inplace=True)
df["order_delivered_customer_date"].fillna(df["order_estimated_delivery_date"], inplace=True)

#dropping order delivery carrier date
df.drop(labels='order_delivered_carrier_date',axis=1,inplace=True)


# In[13]:


# Handling missing values of numerical features
df['product_weight_g'].fillna(df['product_weight_g'].median(),inplace=True)
df['product_length_cm'].fillna(df['product_length_cm'].median(),inplace=True)
df['product_height_cm'].fillna(df['product_height_cm'].median(),inplace=True)
df['product_width_cm'].fillna(df['product_width_cm'].median(),inplace=True)


# In[14]:


#Handling missing values of text column
print("Percentage of null reviews :",(df.review_comment_message.isnull().sum()/len(df))*100 ,"%")
# filling null value of review comments with no_review
df['review_comment_message'].fillna('nao_reveja',inplace=True)


# In[15]:


#Data Dedublicate
duplicate_rows = df[df.duplicated(['order_id','customer_id','order_purchase_timestamp','order_delivered_customer_date','customer_unique_id','review_comment_message'])]
duplicate_rows.head()


# In[16]:


#Droping duplicate rows
df= df.drop_duplicates(subset={'order_id','customer_id','order_purchase_timestamp','order_delivered_customer_date'}, keep='first', inplace=False)
df=df.reindex()
df.head()


# In[17]:


print("Number of rows after dedublication:",len(df))
print("Number of columns after deduplication:",len(df.columns))


# # Data Analysis

# In[18]:


# all time stamps are in object dtype as observed above converting it into dataetime 
df[['order_purchase_timestamp','order_approved_at','order_delivered_customer_date','order_estimated_delivery_date',]]=df[['order_purchase_timestamp',
       'order_approved_at','order_delivered_customer_date','order_estimated_delivery_date']].apply(pd.to_datetime)


# In[19]:


df.info()


# # Observation(s):
# 
# the final merged dataset has no null values.
# total number of columns is 32.
# 
#   dtype           |   number of columns
#   ----------------|-------------------
#   datetime64      |      4
#                   |
#   float64(10)     |      10
#                   |
#   int64           |      5
#                   |
#   object          |      13  
#   

# In[20]:


df.describe()


# # Observation(s):
# 
# We can observe from the above table except customer_zip_code_prefix, order_item_id and review_score features we have 12 numerical features in our final dataset.
# 
# Also,We can observe the statistics like percentile values , mean and standard deviation values, count , min and max of the numerical featues.For payment_value, the maximum payment value of an order is 13664 Brazilian real.
# 
# For the price and freight value of an order. The maximum price of an order is 6735 while max freight is around 410 Brazilian real. The average price of an order is around 125 Brazilian real and frieght value is around 20 Brazilian real. The order with minimum price of 0.85 Brazilian real have been made.
# 
# Similarly, we can observe the other features further we will see the distribution of these features and see how they are helping in classifying the class labels and find other insights.

# In[21]:


# checking the review score 
df.review_score.value_counts()


# In[22]:


def partition(x):
    if x < 3:
        return 0
    return 1
df['review_score']=df['review_score'].map(lambda cw : partition(cw) ) 
    
# checking the review score now
df.review_score.value_counts()


# In[23]:


#counting the review score with 1 and 0
y_value_counts = df.review_score.value_counts()

#calculating the percentage of each review type
print("Total Positive Reviews :", y_value_counts[1], ", (", (y_value_counts[1]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")
print("Total Negative Reviews :", y_value_counts[0], ", (", (y_value_counts[0]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")
print('\n')

#plotting pie chart
get_ipython().run_line_magic('matplotlib', 'inline')
labels = ['Positive','Negative']
sizes = [82895,13621]
plt.figure(figsize=(15,8))
plt.pie(sizes, explode=[0.2, 0] ,labels=labels,colors=('y','g'),autopct='%1.1f%%',shadow=True, startangle=0,radius=1,textprops={'fontsize': 15},frame=False, )
plt.axis('equal') 
plt.title('Pie Chart for review score',color ='black')
plt.show()


# # Observation(s):
# 
# We can observe from the above plots 85.5% of the total reviews are positive i.e. 1 and only 14.5% reviews are negative i.e. which means that the given data set is imbalanced dataset.

# In[24]:


#Correlation matrix 
corr_matrix = df.corr()


# In[25]:


plt.figure(figsize=(18,8))
sns.set(font_scale=1)
cmap = sns.light_palette("#2f3b39",as_cmap=True)
sns.heatmap(corr_matrix, cmap='YlGnBu',annot=True)
plt.title("  Correlation Matrix of the features",fontsize=20)
plt.savefig('plot16.png', dpi=300, bbox_inches='tight')
plt.show()


# # Observation(s):
# 
# There is a strong positive correlation between: (payment_value and price), (product_weight_g and freight_value also with product_width_cm), (product_length_cm and product_width_cm), (product_height_cm and product_weight_g).

# In[26]:


#finding corr- values of the features with review_score
corr_matrix["review_score"].sort_values(ascending=False)


# In[27]:


#checking unique ids
print("Total number of unique seller_id:",len((df.seller_id).unique()))
print("Total number of unique product_id:",len((df.product_id).unique()))
print("Total number of unique customer_id:",len((df.customer_unique_id).unique()))


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(8,6))
sns.set_style("whitegrid")
x = ['seller_id','product_id','customer_id']
y = [3017,30905,93396]
 
# Plot the bar graph
plot = plt.bar(x, y, color = ('yellow','green','#2e4884'),alpha=0.7)

for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom')
 

plt.title("Total Unique Ids",color='Black')
plt.xlabel("Different Unique_Ids",color='Black')
plt.ylabel("No of Unique_Ids",color='Black')
plt.show()


# # observation:
# 
# After comparing different ids it can be observed that the highest number of unique id is of customers and least is from sellers

# # Uivariate Analysis: payment type

# In[29]:


df.groupby('payment_type').size()


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))

sns.set_style("whitegrid")
x = ['debit_card','voucher','boleto','credit_card']
y = [1484,2578,19203,73251]
 
# Plot the bar graph
plot = plt.bar(x, y, color = ('yellow','green','red','#2e4884'),alpha=0.7,edgecolor='black')

for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom')
 

plt.title("Total payment_type",color='Blue')
plt.xlabel("payment_type",color='Black')
plt.ylabel("Quantity",color='Black')
plt.show()


plt.figure(figsize=(10,8))

x1 = ['debit_card','voucher','boleto','credit_card']
y1= [ 1484,2578,19203,73251]
explode = (0.2, 0.2, 0.2,0.2)  
plt.pie(y1, explode=explode, labels=x1 , colors=('y','g','r','#2e4884'), autopct='%1.1f%%',shadow=False, startangle=0,radius=5,frame=False,textprops={'fontsize': 15})
plt.axis('equal') 

plt.show()


# ## Obsrervation(s):
# 
# from the above plots we can observe that most of the orders are paid using credit card and the second most used payment method is boleto.
# 
# The percentage of each mode of payment is shown in pie chart which shows amonst the all payments made by the user the credit card is used by 75.9% of the users, baleto is used by 19.9% of the user and 2.7% of the user used voucher and 1.5% debit card.

# In[31]:


temp = pd.DataFrame(df.groupby('payment_type')['review_score'].agg(lambda x: x.eq(1).sum())).reset_index()

# Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039
temp['total'] = list(pd.DataFrame(df.groupby('payment_type')['review_score'].agg([('total','count'),('Avg','mean')]))['total'])
temp['Avg']   = list(pd.DataFrame(df.groupby('payment_type')['review_score'].agg([('total','count'),('Avg','mean')]))['Avg'])
#sorting dataframe
temp = temp.sort_values(by=['total'], ascending=True)


# In[32]:


#Simplifing the plots using pareto plots
def pareto_plot(df, x=None, y=None, title=None, show_pct_y=False, pct_format='{0:.0%}'):
    xlabel = x
    ylabel = y
    tmp = df.sort_values(y, ascending=False)
    x = tmp[x].values
    y = tmp[y].values
    weights = y / y.sum()
    cumsum = weights.cumsum()

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(x, y,color='#2e4884',edgecolor='black',alpha=0.9)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = ax1.twinx()
    ax2.plot(x, cumsum, '-ro', alpha=0.5,color='black')
    ax2.set_ylabel('', color='r')
    ax2.tick_params('y', colors='r')
    
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    # hide y-labels on right side
    if not show_pct_y:
        ax2.set_yticks([])
    
    formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], cumsum[i]),fontsize=15)    
    
    if title:
        plt.title(title,color='dimgrey',fontsize=15)
    
    plt.tight_layout()
    plt.show()


# In[33]:


pareto_plot(temp,x='payment_type',y='total',title="Pareto Plot of counts of each payment type")


# # observation
# we can observe from the above plots that 96 % of the user used credit card and boleto.With credit card , boleto and voucher it covers 98% of users. Now let us see,how it is related with the target variable i.e review score.Or we can say 98% chance that the customer will use credit_card or boleto or voucher.

# In[34]:


#how this categorical feature related with our target variable
plt.figure(figsize=(10,6))
p1=plt.bar(temp.payment_type,temp.total,color='yellow',alpha=0.8)
p2=plt.bar(temp.payment_type,temp.review_score,color='#2e4884',alpha=0.9)
plt.title('Payment Types and user_counts',fontsize=15,color='black')

plt.xlabel('payment_types',fontsize=14)
plt.ylabel('Total',fontsize=14)
plt.legend((p1[0], p2[0]), ('total_reviews', 'positive_review by users'))

plt.show()


# # Observation(s):
# 
# We can observe from the above stacked plot that most of the customer who used credit card have given positive reviews.Also, for the boleto, voucher and the debit card user it is same.From this we can conclude that this can be our important categorical feature for the problem.

# In[35]:


# State with the consumers count
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
ax = df.customer_state.value_counts().sort_values(ascending=False)[0:15].plot(kind='bar', color = '#2e4884', alpha=0.9)
ax.set_title("Top consumer states of Brazil",color = '#2e4884',alpha=0.9)
ax.set_xlabel("States",color = '#2e4884',alpha=0.9)
plt.xticks(rotation=35)
ax.set_ylabel("No of consumers",color = '#2e4884',alpha=0.9)
plt.show()


# In[36]:


c = df.groupby('customer_state').size()
print(c)


# In[37]:


df.customer_state.count


# In[38]:


p = 100 * (c)/96516 
print(p)


# # Observation(s):
# 
# 41.99% of total consumers are from the SP, 12.85 % are from RJ and 11.7 % are from MG which means most of consumers are from these states.Let us see what type of reviews are given from the consumer of these states.

# In[39]:


#stacked bar plots 
def stack_plot(data, xtick, col2, col3='total'):
    ind = np.arange(data.shape[0])
    
    plt.figure(figsize=(20,5))
    p1 = plt.bar(ind, data[col3].values,color = 'Yellow',alpha=0.5)
    p2 = plt.bar(ind, data[col2].values,color= '#2e4884',alpha=0.9)

    plt.ylabel('Reviews')
    plt.xlabel('States')
    plt.title('% of review_score  ')
    plt.xticks(ind-0.1, list(data[xtick].values), rotation=0)
    plt.legend((p1[0], p2[0]), ('total_reviews', 'positive_review'))
    plt.show()


# In[40]:


# Count number of zeros in dataframe 
temp_1 = pd.DataFrame(df.groupby('customer_state')['review_score'].agg(lambda x: x.eq(1).sum())).reset_index()


temp_1['total'] = list(pd.DataFrame(df.groupby('customer_state')['review_score'].agg([('total','count'),('Avg','mean')]))['total'])
temp_1['Avg']   = list(pd.DataFrame(df.groupby('customer_state')['review_score'].agg([('total','count'),('Avg','mean')]))['Avg'])
temp_1= temp_1.rename(columns={'review_score':'positive_review'})
temp_1= temp_1.sort_values(by=['total'], ascending=False)


# In[41]:


temp_1


# In[42]:


stack_plot(temp_1,'customer_state',col2='positive_review', col3='total')


# # Observation(s):
# 
# From the avove stack plot of reviews per state we can conclude that most of consumers from each state has given positive reviews.In SP state from the total reviews of 40536 , 35697 reviews are positive reviews and for RJ state 9907 reviews are positive from the total reviews 12410.The consumer_state can be our important feature for the problem.

# In[43]:


# Univariate Analysis: product_category_name_english
# State with the consumers count
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
ax = df.product_category_name_english.value_counts().sort_values(ascending=False)[0:15].plot(kind='bar', color = '#2e4884',alpha=0.9)
ax.set_title("Top selling product categories")
ax.set_xlabel("Products")
plt.xticks(rotation=90)
ax.set_ylabel("No of Orders")
plt.show()


# In[44]:


temp_2 = pd.DataFrame(df.groupby('product_category_name_english')['review_score'].agg(lambda x: x.eq(1).sum())).reset_index()

temp_2['total'] = list(pd.DataFrame(df.groupby('product_category_name_english')['review_score'].agg([('total','count'),('Avg','mean')]))['total'])
temp_2['Avg']   = list(pd.DataFrame(df.groupby('product_category_name_english')['review_score'].agg([('total','count'),('Avg','mean')]))['Avg'])
temp_2 = temp_2.sort_values(by=['total'], ascending=True)
temp_2


# In[45]:


plt.figure(figsize=(22,18))
plt.barh(temp_2.product_category_name_english,temp_2.total,color='Yellow',alpha=0.6)
plt.barh(temp_2.product_category_name_english,temp_2.review_score,color='#2e4884',alpha=0.9)
plt.title('Top Selling Product Categories in Brazilian E-Commerce (2016-2018)',fontsize=22,color='#2e4884',alpha=0.9)
plt.ylabel('product_category_name_english',fontsize=14)
plt.xlabel('Total',fontsize=14)
plt.savefig('plot14.png', dpi=480, bbox_inches='tight')
plt.show()


# Observation(s):
# 
# From the first plot titled Top selling product categories we can conclude that most ordered products is from bed_bath_table category ,health beauty and sports_leisure between 2016 and 2018.The least ordered products are from security_and_services.
# 
# The second plot is stack plot which shows the total reviews and the reviews with positive sense.from this plot we can conclude that most of the reviews for the product category bed_bath_table are positive and it is same for the other product categories.This can be our important categorical feature for the problem.

# # frequency of orders Vs Number of Consumers

# In[46]:


# plotting frequency orders vs  the number of consumers 
plt.figure(figsize=(14,8))

#counting the consumers and converting it into percentage to visualize the distribution properly
num_orders=df['customer_unique_id'].value_counts().value_counts()/df.shape[0]*100
num_orders=num_orders.reset_index()
#renaming the columns
num_orders.rename(columns={'index':'number of orders', 'customer_unique_id':'log percentage of customers'},inplace=True)

#plotting bar plot
sns.barplot(data=num_orders,x='number of orders',y='log percentage of customers',palette='Blues')
plt.yscale('log') #log scale
plt.title('Number of orders per customer',color='dimgrey')


# Observation(s):
# 
# Most of the consumer given order of any products only for one times and few consumers are also present who oredred products more than 15 times. From this we can say order frequecy can be used as important feature for the problem.

# # Order_status

# In[47]:


os=df.groupby('order_status').size()
print(os)


# In[48]:


d = 100 * (os)/96516 
print(d)


# In[49]:


plt.figure(figsize=(10,8))

sns.set_style("whitegrid")
x = ['approved','canceled','delivered','invoiced','processing','shipped','unavailable']
y = [0.002072,0.443450,97.898794,0.310829,0.294252,1.044386,0.006217]
 
# Plot the bar graph
plot = plt.bar(x, y, color = ('#2e4884'),alpha=0.7,edgecolor='black')

for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom')
 

plt.title("order_status",color='Blue')
plt.xlabel("order_status",color='Black')
plt.ylabel("percentage",color='Black')
plt.show()


# Observation(s):
# 
# The data shows that 97.8% of the orders of status delivered and remaining precentage are with status shipped,canceled,invoiced,processing,unavailable and approved.

# In[50]:


df_new = df.groupby(['order_status', 'review_score',]).size()


# In[51]:


df_new


# In[52]:


rper = 100 * (df_new)/96516 
print(rper)


# In[53]:


plt.figure(figsize=(14,8))
X = ['approved','canceled','delivered','invoiced','processing','shipped','unavailable']
negative = [0.001036,0.367815,12.484977,0.255916,0.272494,0.725268,0.005180]
positive = [0.001036,0.075635,85.413817,0.054913,0.021758,0.319118,0.001036]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, negative, 0.4,label = 'negative')
plt.bar(X_axis + 0.2, positive, 0.4,label = 'positive')
for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom')
plt.xticks(X_axis, X)
plt.xlabel("order_status")
plt.ylabel("Percentage")
plt.title("Order_status with % of Reviews")
plt.legend()
plt.show()


# plot shows that for the order_status delivered most of the order are with positive reviews i.e 85% and only 12.4% are negative reviews.

# # Analysis on different Timestamps

# In[54]:


#calulating number of days for the data is taken
print(df.order_approved_at.max() - df.order_approved_at.min(), ' from ', 
      df.order_approved_at.min(), ' to ', df.order_approved_at.max())


# In[55]:


# Extracting attributes for purchase date - Year and Month
df['order_purchase_year'] = df['order_purchase_timestamp'].apply(lambda x: x.year) 
df['order_purchase_month'] = df['order_purchase_timestamp'].apply(lambda x: x.month) 
df['order_purchase_month_name'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%b'))
df['order_purchase_year_month'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m'))
df['order_purchase_date'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m%d'))
df['order_purchase_month_yr'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime("%b-%y"))

# Extracting attributes for purchase date - Day and Day of Week
df['order_purchase_day'] = df['order_purchase_timestamp'].apply(lambda x: x.day)
df['order_purchase_dayofweek'] = df['order_purchase_timestamp'].apply(lambda x: x.dayofweek)
df['order_purchase_dayofweek_name'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%a'))

# Extracting attributes for purchase date - Hour and Time of the Day
df['order_purchase_hour'] = df['order_purchase_timestamp'].apply(lambda x: x.hour)
hours_bins = [-0.1, 6, 12, 18, 23]
hours_labels = ['Dawn', 'Morning', 'Afternoon', 'Night']
df['order_purchase_time_day'] = pd.cut(df['order_purchase_hour'], hours_bins, labels=hours_labels)

# New DataFrame after transformations
df.head()


# In[56]:


plt.figure(figsize=(15,6))
sns.lineplot(data=df['order_purchase_year_month'].value_counts().sort_index(), 
             color='black', linewidth=2)
plt.title('Evolution of Total Orders in Brazilian E-Commerce', size=14, color='dimgrey')
plt.xticks(rotation=90)
plt.show()


# # Observation(s):
# 
# from the above plot we can observe that the number of purchase is increasing from 201609 to 201711(highest) and then decreases for a short sapn which means the either orders from the older customers are increasing or the number of consumers are increasing.

# In[57]:


df_month = pd.DataFrame()
df_month['date'],df_month['review_score']= list(df.order_approved_at),list(df.review_score)
df_month=df_month.dropna()
df_month = df_month.sort_values(by=['date'])


# In[58]:


df_month['monthcount'] = list(df_month.date.apply(lambda x: x.strftime("%b-%y")))
#plotting number of orders per month-year
plt.figure(figsize=(18,6))
g = sns.countplot(x=df_month.monthcount,data=df_month,color='grey',edgecolor='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Month-Year')
g.set_ylabel('Orders Count')
plt.title('Number of orders per month-year', size=14, color='dimgrey');


# In[59]:


#plotting number of positive and negative reviews per month-year
plt.figure(figsize=(18,6))
g = sns.countplot(x=df_month.monthcount,hue='review_score',data=df_month,palette=['grey','#425a90'],edgecolor='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Month-Year')
g.set_ylabel('Orders Count')
plt.title('Number of positive and negative reviews per month-year', size=14, color='dimgrey');


# Observation(s):
# 
# From the first plot titled Number of orders per month-year show the total number order received per month on each each between 2016 and 2018.It can be observeed that in the month of November in the year 2017 the highest number of orders are received which is more the 7000 approx. and the leat number of orders are received in the month of Dec in the year 2016.
# 
# The second plot shows the total number of positive and negative reviews given for each order per month-year.It can observed that most the orders have given positive reviews.From the above plot we observed that on NOV-17 the highest orders were received but from the second plot we can see the highest positive reviews were given on May-18

# In[60]:


#code source: https://www.kaggle.com/thiagopanini/e-commerce-sentiment-analysis-eda-viz-nlp
fig = plt.figure(constrained_layout=True, figsize=(13, 10))

# Axis definition
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[0, :])
ax3 = fig.add_subplot(gs[1, 1])

# Barchart - Total Reviews by time of the day
single_countplot(df, x='order_purchase_time_day', ax=ax1, order=False, palette=['grey','#2e4884'],hue='review_score')
ax1.set_title('Total Reviews by Time of the Day', size=14, color='dimgrey', pad=20)

# Barchart - Total Reviews by month
single_countplot(df, x='order_purchase_month_name', ax=ax2, order=False, palette=['grey','#2e4884'],hue='review_score')

ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep','Oct','Nov','Dec'])
ax2.set_title('Total Reviews by Month', size=14, color='dimgrey', pad=20)

single_countplot(df, x='order_purchase_dayofweek', ax=ax3, order=False, palette=['grey','#2e4884'],hue='review_score')
weekday_label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax3.set_xticklabels(weekday_label)
ax3.set_title('Total Reviews by Day of Week', size=14, color='dimgrey', pad=20)

plt.savefig('plot14.png', dpi=300, bbox_inches='tight')
plt.tight_layout()

plt.show()


# In[61]:


#ploting plot for the Total Number orders based on the Total delivery Time(Days)
#https://stackoverflow.com/questions/60229375/solution-for-specificationerror-nested-renamer-is-not-supported-while-agg-alo
df['day_to_delivery']=((df['order_delivered_customer_date']-df['order_purchase_timestamp']).dt.days)


# In[62]:


df_dev = pd.DataFrame()
df_dev['day_to_delivery'],df_dev['review_score']= list(df.day_to_delivery),list(df.review_score)
df_dev=df_dev.dropna()


# In[63]:


plt.figure(figsize=(22,6))
plt.title('Order Counts Based on Total delivery Time(in Days)', color='dimgrey')
g = sns.countplot(x=df_dev.day_to_delivery,data=df_dev,color='gray')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Total Days')
g.set_ylabel('Orders Count');


# Observation(s):
# 
# It can be observed that the maximum number of orders are delivered in 7 days few orders are also delivered in more than 30 days.The total deliver time can be a new feature to solve this problem.

# # Univariate Analysis on Numerical Features
# Price

# In[64]:


import seaborn as sns

plt.figure()
sns.set_style("whitegrid")
ax = sns.FacetGrid(df, hue="review_score", height=5,aspect=2.0,palette=['#2e4884','black'])
ax = ax.map(sns.distplot, "price").add_legend();
plt.title('Distribution of product price per class')
plt.show()


# The above distribution plot shows the distribution of price for both the postive and negative classes. We can observe that there is almost completely overlap of both the distribution for positive and negative class which suggests that it is not possible to classify them based only on price feature.
# 
# freight_value

# In[65]:


# plotting distributions of freight_value per class
plt.figure()
#sns.set_style("whitegrid")
ax = sns.FacetGrid(df, hue="review_score", height=5,aspect=2.0,palette=['#000080','black'])
ax = ax.map(sns.distplot, "freight_value").add_legend();
plt.title('Distribution of freight_value per class')
plt.show()


# The above distribution plot shows the distribution of freight_value for both the postive and negative classes. We can observe that there is almost completely overlap of both the distribution for positive and negative class which suggests that it is not possible to classify them based only on freight_value feature.
# 
# product_height_cm

# In[66]:


# plotting distributions of product_height_cm per class
sns.set_style("whitegrid")
ax = sns.FacetGrid(df, hue="review_score", height=5,aspect=2.0,palette=['#2e4884','black'])
ax = ax.map(sns.distplot, "product_height_cm").add_legend();
plt.title('Distribution of product_height_cm per class')
plt.show()


# The above distribution plot shows the distribution of product_height_cm for both the postive and negative classes. We can observe that most of the product has height less than 20 .Also, there is almost completely overlap of both the distribution for positive and negative class which suggests that it is not possible to classify them based only on product_height_cm feature.
# 
# product_weight_g

# In[67]:


# distriution plot of product_weight_g
plt.figure()
sns.set_style("whitegrid")
ax = sns.FacetGrid(df, hue="review_score", height=5,aspect=2.0,palette=['#2e4884','black'])
ax = ax.map(sns.distplot, "product_weight_g").add_legend();
plt.title('Distribution of product_weight_g per class')
plt.show()


# The above distribution plot shows the distribution of product_weight_g for both the postive and negative classes. We can observe that most of the product has weight less than 5000 gm .Also, there is almost completely overlap of both the distribution for positive and negative class which suggests that it is not possible to classify them based only on product_weight_g feature.
# 
# payment_value

# In[68]:


# distriution plot of payment_value
plt.figure()
sns.set_style("whitegrid")
ax = sns.FacetGrid(df, hue="review_score", height=5,aspect=2.0,palette=['#2e4884','black'])
ax = ax.map(sns.distplot, "payment_value").add_legend();
plt.title('Distribution of payment_value per class')
plt.show()


# The above distribution plot shows the distribution of payment_value for both the postive and negative classes. We can observe that there is almost completely overlap of both the distribution for positive and negative class which suggests that it is not possible to classify them based only on payment_value feature.
# 
# Box Plot

# In[69]:


import matplotlib.pyplot as plt
 
plt.figure(figsize=(14,6))
 
box_plot_data=[df.product_length_cm,df.product_height_cm,df.product_width_cm]
plt.boxplot(box_plot_data,labels=['product_length_cm','product_height_cm','product_width_cm'],vert=False)
plt.title("Box Plots of Product Dimensions")
plt.savefig('plot24.png', dpi=400, bbox_inches='tight')
plt.show()


# In[70]:


import matplotlib.pyplot as plt
 
plt.figure(figsize=(20,6))
 
box_plot_data=[df.payment_value,df.price]
plt.boxplot(box_plot_data,labels=['payment_value','price'],vert=False)
plt.title("Box Plots of Different Prices")
plt.savefig('plot25.png', dpi=400, bbox_inches='tight')
plt.show()


# The above box plots are showing the distribution of the numerical features product_width_cm, product_height_cm and product_width_cm. These features are overlapping each other.
# 
# Now, let us go and do some bivariate analysis and see if we use more than one feature at time, can we come with something to classify these features.

# # Bivariate Analysis

# In[71]:


#Distribution of price vs freight_value per class
plt.figure(figsize=(8,5))
sns.set_style("whitegrid")
ax = sns.scatterplot(x='price',y='freight_value', data = df, hue="review_score",palette=['#2e4884','grey'])
plt.title('Distribution of price vs freight_value per class')
plt.show()


# In[72]:


# Distribution of price vs freight_value per class
plt.figure(figsize=(8,5))
sns.set_style("whitegrid")
ax = sns.scatterplot(x='price',y='product_weight_g', data = df, hue="review_score",palette=['#2e4884','grey'])
plt.title('Distribution of price vs product_weight_g per class')
plt.show()


# Obervation(s):
# 
# * From the above two scatter plots titled `Distribution of price vs freight_value per class` and `Distribution of price vs freight_value per class` respectively, It is very hard to say anything about the reviews on the basis of these plot as data-points are not seperable based on reviews these are completely mixed data.

# In[73]:


# https://seaborn.pydata.org/generated/seaborn.pairplot.html
# pair plot
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df[['product_photos_qty','product_name_lenght','product_description_lenght','review_score']],hue='review_score',palette=['#2e4884','grey'])
g.savefig("pairplot1.png")


# Observation(s):
# 
# The pair plot shown above for the features product_photos_qty, product_name_length,product_description_length as these have negative correlation values with the review_score column.All the scatter plots between the features are completely mixed up not separable on the basis of reviews.We can say that none of these features are helpful for the classification.

# # Multivariate Analysis

# Evolution of price and the total orders per month

# In[74]:


df_mm=df[['order_purchase_month_name','price']].groupby('order_purchase_month_name').sum()


# In[75]:


pi = list(df_mm['price'])
li = list(df_mm.index)
#dict of months and price value
res = {li[i]: pi[i] for i in range(len(li))}


# In[76]:


from collections import OrderedDict
mnths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep','Oct','Nov','Dec']
weeks=['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
res = dict(OrderedDict(sorted(res.items(),key =lambda x:mnths.index(x[0]))))#sorting by month
print(res)


# In[77]:


temp_3= pd.DataFrame(df.groupby('order_purchase_month_name')['review_score'].agg(lambda x: x.eq(1).sum())).reset_index()

# Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039


temp_3['total'] = list(pd.DataFrame(df.groupby('order_purchase_month_name')['review_score'].agg([('total','count'),('Avg','mean')]))['total'])
temp_3['Avg']   = list(pd.DataFrame(df.groupby('order_purchase_month_name')['review_score'].agg([('total','count'),('Avg','mean')]))['Avg'])
temp_3= temp_3.sort_values(by=['total'], ascending=True)


# In[78]:


rem = {list(temp_3.order_purchase_month_name)[i]: list(temp_3.total)[i] for i in range(len(temp_3))}
rem = dict(OrderedDict(sorted(rem.items(),key =lambda x:mnths.index(x[0]))))
print(rem)


# In[79]:


sns.set_style("whitegrid")


fig, ax1 = plt.subplots()

color = 'grey'
ax1.set_xlabel('Month')
ax1.set_ylabel('price', color=color)
ax1.plot(list(res.keys()),list(res.values()), color=color)
ax1.plot(list(res.keys()),list(res.values()),'C0o', alpha=0.5,color='grey')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = '#2e4884'
ax2.set_ylabel('orders', color=color)  # we already handled the x-label with ax1
ax2.plot(list(res.keys()),list(rem.values()), color=color)
ax2.plot(list(res.keys()),list(rem.values()),'C0o', alpha=0.5,color='#2e4884')
ax2.tick_params(axis='y', labelcolor=color)
#creating  points 


fig.tight_layout( )  # otherwise the right y-label is slightly clipped
plt.show()


# from the above plots we can observe that there is same pattern of total sales and the total order per month between 016 and 2018.

# # RFM -Analysis

# In[87]:


PRESENT = datetime(2018,9,3)
rfm= df.groupby('customer_unique_id').agg({'order_purchase_timestamp': lambda date: (PRESENT - date.max()).days,
                                        'order_id': lambda num: len(num),
                                        'payment_value': lambda price: price.sum()})
rfm.columns=['recency','frequency','monetary']
rfm['recency'] = rfm['recency'].astype(int)
rfm['frequency'] = rfm['frequency'].astype(int)
rfm['monetary'] = rfm['monetary'].astype(float)


# In[88]:


rfm.head()


# In[89]:


# Plot RFM distributions
plt.figure(figsize=(12,10))
# Plot distribution of R
plt.subplot(3, 1, 1); sns.distplot(rfm['recency'],color='black')
# Plot distribution of F
plt.subplot(3, 1, 2); sns.distplot(rfm['frequency'],color='black')
# Plot distribution of M
plt.subplot(3, 1, 3); sns.distplot(rfm['monetary'],color='black')
# Show the plot
plt.show()


# Observation(s)
# 
# There are three density plots of recency, frequency and monetary are plotted.From the first plot of recency we can observe that most of the users stayed with olist for long duration which is positive thing but order frequency is less.
# 
# from the second plot of frequency most number of transaction or order is less than 5. from the third plot of monetary the maximum amount spend over the given very period is seems to less than 1500 approx.

# In[90]:


# Create labels for Recency and Frequency
def partition(x):
    if x < 10:
      return 1
    if 10<=x<=35:
      return 2
    if 35<x<=50:
      return 3
    if 50<x<=75:
      return 4      

rfm['f_quartile']=rfm['frequency'].map(lambda cw : partition(cw) ) 
    
# checking the review score now
rfm.f_quartile.value_counts()
r_labels = range(4, 0, -1);m_labels= range(1,5)

rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, r_labels)
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, m_labels)


# In[91]:


rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm.head()


# In[92]:


rfm_count_unique = rfm.groupby('RFM_Score')['RFM_Score'].nunique()
print(rfm_count_unique.sum())
rfm['RFM_Score_s'] = rfm[['r_quartile','f_quartile','m_quartile']].sum(axis=1)
print(rfm['RFM_Score_s'].head())


# In[93]:


# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score_s'] >= 9:
        return 'Can\'t Loose Them'
    elif ((df['RFM_Score_s'] >= 8) and (df['RFM_Score_s'] < 9)):
        return 'Champions'
    elif ((df['RFM_Score_s'] >= 7) and (df['RFM_Score_s'] < 8)):
        return 'Loyal'
    elif ((df['RFM_Score_s'] >= 6) and (df['RFM_Score_s'] < 7)):
        return 'Potential'
    elif ((df['RFM_Score_s'] >= 5) and (df['RFM_Score_s'] < 6)):
        return 'Promising'
    elif ((df['RFM_Score_s'] >= 4) and (df['RFM_Score_s'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'
# Create a new variable RFM_Level
rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)
# Print the header with top 5 rows to the console
rfm.head()


# In[94]:


# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = rfm.groupby('RFM_Level').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': ['mean', 'count']
}).round(1)
# Print the aggregated dataset
print(rfm_level_agg)
rfm_level_agg.columns = rfm_level_agg.columns.droplevel()


# In[96]:


import squarify

rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=rfm_level_agg['Count'], 
              label=['Can\'t Loose Them',
                     'Champions',
                     'Loyal',
                     'Needs Attention',
                     'Potential', 
                     'Promising', 
                     'Require Activation'], alpha=.9,color=['#f0f0f0','#d2d2d2','#b4b4b4','#a5a5a5','#969696','#425a90','#2e4884'])
plt.title("RFM Segments",fontsize=18)
plt.axis('off')
plt.show()


# In[97]:


rfm.head()


# Observation(s):
# 
# Based on the RFM_Score_s all customers are categorised into 7 categories :
# 'Can\'t Loose Them' ====  RMF_Score_s  ≥  9
# 'Champions' ==== 8 ≤ RMF_Score_s < 9
# 'Loyal' ==== 7 ≤ RMF_Score_s <8
# 'Needs Attention' ==== 6 ≤ RMF_Score_s <7
# 'Potential' ==== 5 ≤ RMF_Score_s < 6
# 'Promising' ==== 4 ≤ RMF_Score_s < 5 
# 'Require Activation' RMF_Score_s <4
# From the above square plot the highest percentage of customers lie within area of category potential.Few areas also there with colored in blue scale which show the percentage of comsumers which requries more attention so that they can retain in olist.
# 
# We can use either RMF_Score_s or RMF_Level as feature to solve this problem.

# In[98]:


#saving file 

rfm.to_pickle('rfm.pkl')
df.to_pickle('final.pkl')


# The target variable/class-label is imbalanced.We should be carefull while choosing the performance metric of the models.
# 
# From the Correlation matrix we found that there is a strong positive correlation between: (payment_value and price), (product_weight_g and freight_value also with product_width_cm), (product_length_cm and product_width_cm), (product_height_cm and product_weight_g).But most of the features doesnot seems to be helpful for the classification.
# 
# From the univariate analysis of payment_type we observed that 96 % of the user used credit card and boleto and concluded that this can be our important feature.
# 
# Also,from the univariate analysis of consumer_state we found that 42% of total consumers are from the SP(São Paulo), 12.9 % are from RJ(Rio de Janeiro) and 11.7 % are from MG(Minas Gerais).
# 
# After analysing the product_category feature we observed that the most ordered products is from bed_bath_table category ,health beauty and sports_leisure between 2016 and 2018.The least ordered products are from security_and_services.
# 
# The different timestamps seems to be important features as many new features can be explorated from these.we observed within 2016-18 the total number of order received is incresing till 2017-11 and after that there small decrement.from the month, day and time we observed the most number of orders are received in the month of feb , on monday and afternoon time.
# 
# The numerical features like price, payment_value, freight_value,product_height_cm,product_length_cm doesnot seems to be helpful for this classification problem as observed from univariate and bivarate analysis.
# 
# As review_message can be important feature for this problem,basic analysis of the text is done and found that the most frequent words are 'o', 'e','produto','a' e.t.c.
# 
# RMF Analyis is also done to understand weather new features can be created from this or not and we found that one numerical feature or categorical feature can be extracted from this.

# In[ ]:




