import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
 



passenger_df=pd.read_csv('titanic.csv')
passenger_df = passenger_df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
print ('Number of male passengers: ', len(passenger_df.groupby('Sex').groups['male']))
print ('Number of female passengers: ', len(passenger_df.groupby('Sex').groups['female']))
male_passenger = passenger_df[passenger_df['Sex']== 'male']
female_passenger = passenger_df[passenger_df['Sex']== 'female']
kid_passenger = passenger_df[passenger_df['Age'] < 16]
male_kid_passenger = kid_passenger[kid_passenger['Sex'] == 'male']
female_kid_passenger = kid_passenger[kid_passenger['Sex'] == 'female']
print(male_passenger)
adult_male_passenger = male_passenger.drop(male_kid_passenger.index[:])
adult_female_passenger = female_passenger.drop(female_kid_passenger.index[:])
print ('Number of all passengers:', len(passenger_df))
print ('Number of male passengers:', len(male_passenger))
print ('Number of female passengers:', len(female_passenger))
print ('Number of adult male passengers:', len(adult_male_passenger))
print ('Number of adult female passengers:', len(adult_female_passenger))
print ('Number of kid passengers:', len(kid_passenger)) 
x = [len(male_passenger), len(female_passenger)]
label = ['Male', 'Female']
plt.pie(x, labels = label, autopct = '%1.01f%%')
plt.title('Distribution of passengers by sex')
plt.show()
def age_distribution(x):
    if x>=0 and x <16:
        return 'Child'
    elif x>=16 and x<=24:
        return 'Young'
    else:
        return 'Adult'
passenger_df['Age'].apply(age_distribution).value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Distribution of passengers by age')
plt.show()
print("*******************************************")
print ('Yetişkin Erkek Yolcuların Yas Ortalaması:', adult_male_passenger['Age'].mean())
print ('Yetişkin Kadın Yolcuların Yas Ortalaması:', adult_female_passenger['Age'].mean())
print (' Cocuk Yolcuların Yas Ortalaması:', kid_passenger['Age'].mean())
passenger_df['Pclass'].value_counts()
passenger_df['Pclass'].value_counts().plot(kind='barh', color='green', figsize=[16,4])
plt.xlabel('Frequency')
plt.ylabel('Yolcu Sınıfı')
plt.show()
first_class_passenger = passenger_df[passenger_df['Pclass'] == 1]
second_class_passenger = passenger_df[passenger_df['Pclass'] == 2]
third_class_passenger = passenger_df[passenger_df['Pclass'] == 3]
print(passenger_df['Embarked'].describe())
passenger_df['Embarked'].value_counts().plot(kind='bar')
plt.title('Embarking ports')
plt.ylabel('Frequency')
plt.xlabel('S=Southampton, C=Cherbourg, Q=Queenstown')
plt.show()

print(passenger_df['Survived'].value_counts())
print(passenger_df.groupby('Sex')['Survived'].value_counts())
passenger_df.groupby('Sex')['Survived'].value_counts().plot(kind='bar', stacked=True, colormap='winter')
plt.show()
sex_survived = passenger_df.groupby(['Sex', 'Survived'])
sex_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')
plt.ylabel('Frequency')
plt.title('Survivings by sex')
plt.show()
print ('Mean of survived adult female passengers:', adult_female_passenger['Survived'].mean())
print ('Mean of survived adult male passengers:', adult_male_passenger['Survived'].mean())
class_survived = passenger_df.groupby(['Pclass', 'Survived'])
print(class_survived.size().unstack())
class_survived.size().unstack().plot(kind='bar', stacked=True, colormap='autumn')
plt.xlabel('1st = Upper,   2nd = Middle,   3rd = Lower')
plt.ylabel('Frequency')
plt.title('Survivings by passenger class')
plt.show()
print ('Surviving numbers of male passengers by passenger class: ',
male_passenger.groupby(['Pclass', 'Survived']).size().unstack())
print ('Surviving numbers of female passengers by passenger class:',
female_passenger.groupby(['Pclass', 'Survived']).size().unstack())
fig, axes = plt.subplots(nrows=2, ncols=1)
male_passenger.groupby(['Pclass','Survived']).size().unstack().plot(kind='bar', title='Surviving of male passengers by class',
                                                                    stacked=True, colormap='summer', ax=axes[0])
female_passenger.groupby(['Pclass','Survived']).size().unstack().plot(kind='bar', title='Surviving of female passengers by class',
                                                                      stacked=True, colormap='summer', ax=axes[1])
plt.tight_layout()
plt.show()
without_sibsp_passenger = passenger_df[passenger_df['SibSp']==0]
alone_passenger = without_sibsp_passenger[without_sibsp_passenger['Parch']==0]
print(alone_passenger.head(7))
family_passenger = passenger_df.drop(alone_passenger.index[:])
print(family_passenger.tail(6))
print ('Mean of survived alone passengers:', alone_passenger.groupby('Sex')['Survived'].mean())
print ('Mean of survived passengers with family:', family_passenger.groupby('Sex')['Survived'].mean())
print ('')
print ('')
#Mean of survived alone passengers and passengers with family by passenger class
print ('Mean of survived alone passengers:', alone_passenger.groupby('Pclass')['Survived'].mean())
print ('Mean of survived passengers with family:', family_passenger.groupby('Pclass')['Survived'].mean())
passenger_df['Fare'].fillna(passenger_df['Fare'].dropna().median(), inplace=True)
passenger_df['FareBand'] = pd.qcut(passenger_df['Fare'], 4)
passenger_df['FareBand'].value_counts().sort_values(ascending= False)
passenger_df[['FareBand', 'Survived']].groupby(['FareBand'],
                          as_index=False).mean().sort_values(by='FareBand',
                          ascending=True)
                                                             
##############  Veri Tahmin Etme #############



df = pd.read_csv('titanic.csv')
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)

df=df.dropna()

dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
    
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat((df,titanic_dummies), axis=1)
df = df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)


df['Age'] = df['Age'].interpolate()

X = df.values
y = df['Survived'].values
X = np.delete(X, 1, axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
 
y_pred = regressor.predict(X_test)
print(y_pred)










