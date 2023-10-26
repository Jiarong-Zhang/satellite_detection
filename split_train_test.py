from sklearn.model_selection import train_test_split
import pandas as pd

file = pd.read_csv('bbox.csv')

train, test = train_test_split(file, test_size=0.2, random_state = 2023)

#print(train)
train.to_csv("train.csv")

#print(test)
test.to_csv("test.csv")