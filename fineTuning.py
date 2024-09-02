import pandas as pd

df = pd.read_csv("TrainningData/sms_spam_collection/SMSSpamCollection.tsv", sep="\t", header=None, names=["Label", "Text"])
print(df)
print(df["Label"].value_counts())