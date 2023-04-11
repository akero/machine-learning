import pandas as pd
import matplotlib.pyplot as plt



# Load CSV file into a DataFrame
df = pd.read_csv('Apple.csv')
#print(df.head())
#print(df.info())
#sort db by date and set index to date column

df=df.sort_values('Date')
df=df.set_index('Date')



#split data 80%
split = int(len(df) * 0.8)
train_data = df.iloc[:split]
test_data = df.iloc[split:]

#save to two files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

#visualize data

plt.plot(df['Close'])
plt.title('Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()