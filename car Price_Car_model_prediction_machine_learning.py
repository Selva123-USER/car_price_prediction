import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

df = pd.read_csv("car_model_dataset.csv")
print(df.head(60))

#



df_column_list = df.columns
print(df_column_list)
#
df = df.rename(columns={"CarName":"Name"})
print(df.head(60))

print(df.shape)
#
df_duplicated = df[df.duplicated()]
print("duplicated_rows:",df_duplicated.shape) #0,21 = 21 column-- o duplicate rows

print(df.count())

df = df.drop_duplicates()

# print(df.head(60))
#
#
print(df.count())

print(df.dtypes)

print(df.columns)

df.drop(columns=["Make","Driven_Wheels","Vehicle Style"],inplace=True)

print(df.columns)

print(df.dtypes)

df_category = df["Engine Fuel Type"].unique()
print(df_category)

label_encoding_object  = LabelEncoder()
df["Model"] = label_encoding_object.fit_transform(df[["Model"]])
df["Engine Fuel Type"] = label_encoding_object.fit_transform(df[["Engine Fuel Type"]])
df["Transmission Type"] = label_encoding_object.fit_transform(df[["Transmission Type"]])
df["Market Category"] = label_encoding_object.fit_transform(df[["Market Category"]])
df["Vehicle Size"]  = label_encoding_object.fit_transform(df[["Vehicle Size"]])

print(df.dtypes)

null_value = df.isnull().sum()
print(null_value)

print(df.count())

df_category = df["Engine Fuel Type"].unique()
print(df_category)
#
#
df["Engine HP"] = df["Engine HP"].fillna(df["Engine HP"].mean())
df["Engine Cylinders"] = df["Engine Cylinders"].fillna(df["Engine Cylinders"].mean())
#
# #
df.drop(columns=["Number of Doors","Market Category"],inplace=True)
#
print(df.dtypes)

print(df.count())
# #
plt.boxplot(df["Engine Fuel Type"])
plt.show()
# #
#
print(df.dtypes)


sns.pairplot(df,x_vars=["Model","Engine HP","Vehicle Size"],y_vars="MSRP",aspect=0.5,height=4,kind="scatter")
plt.show()

sns.heatmap(df.corr(),annot=True,cmap="Blues")
plt.show()

df_null = df.isnull().sum()
print(df_null)

q1 = np.quantile(df["Engine Fuel Type"],0.25)
q3 = np.quantile(df["Engine Fuel Type"],0.75)

print(q1,q3)

IQR = q3 - q1
print(IQR)



lower_range = q1-1.5*IQR
higher_range = q3+1.5*IQR
print(lower_range)
print(higher_range)

print(df["Engine Fuel Type"].value_counts())


outlier = df[(df["Engine Fuel Type"]<lower_range) | (df["Engine Fuel Type"]>higher_range)]

print("outliers:\n",outlier)
#
df_remove_outlier = df[(df["Engine Fuel Type"]>= lower_range) & (df["Engine Fuel Type"]<=higher_range)]

print("df_remove_outlier:\n",df_remove_outlier)

plt.boxplot(df_remove_outlier["Engine Fuel Type"])
plt.show()




df_min_max = StandardScaler()
df["MSRP"] = df_min_max.fit_transform(df[["MSRP"]])



print(df["MSRP"])


fig, axes = plt.subplots(3,1,figsize = (10,10))


axes[0].boxplot(df[["Model"]])
axes[1].boxplot(df[["Vehicle Size"]])
axes[2].boxplot(df[["MSRP"]])




plt.tight_layout()
plt.show()


q1 = np.quantile(df["MSRP"],0.25)
q3 = np.quantile(df["MSRP"],0.75)

print(q1,q3)

iqr = q3 - q1
print(iqr)

lower = q1-1.5*iqr
higher = q3+1.5*iqr

print(lower, higher)
#
out = df[(df["MSRP"]<lower) | (df["MSRP"]>higher)]

df_outlier_remove = df[(df["MSRP"]>= lower_range) & (df["MSRP"]<=higher_range)]
print("df_remove_outlier:\n",df_outlier_remove)

plt.boxplot(df_outlier_remove["MSRP"])
plt.show()



x = df[["Engine HP","Engine Cylinders"]]
y = df["MSRP"]

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

model = LinearRegression()
model.fit(x_train,y_train)

print("-------------------x_train,x_test--------------------")

print("---------------x_train-----------------")
print(x_train.head(100))
print("---------------x_test-----------------")
print(x_test.head(100))

print("-------------------y_train,y_test--------------------")
print("-----------------y_train---------------------")
print(y_train.head(100))
print("-----------------y_test---------------------")
print(y_test.head(100))


output = model.predict(x_test)
print("-----------output_prediction----------")
print(output)


df_correlation = df["Engine HP"].corr(df["MSRP"])
print("----------------co_relation-----------------")
print(df_correlation)



# mae = mean_absolute_error(y_test,output)
# print(mae)



mse = mean_squared_error(y_test,output)
print("mean_squared_error:\n",mse)
# #
# print(df.shape)
# #
# #
#





