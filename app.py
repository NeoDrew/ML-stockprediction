import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

sp500 = sp500.drop(["Dividends","Stock Splits"], axis=1)
sp500 = sp500[sp500.index > pd.to_datetime('2024-01-01 00:00:00-05:00').tz_convert('America/New_York')]
sp500["Tomorrow"] = sp500["Close"].shift(-1)

plt.figure(figsize=(6, 3))
plt.plot(sp500.index, sp500["Open"], label='Close')
plt.plot(sp500.index, sp500["Close"], label='Tomorrow', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('S&P 500 Close Price and Tomorrow\'s Price')
plt.legend()
plt.grid(True)
plt.show()

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neural_network import MLPClassifier

# Split the data into training and testing sets
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Define the predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Scale the features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[predictors])
test_scaled = scaler.transform(test[predictors])

# Check the distribution of the target variable
print("Target distribution in training set:")
print(train["Target"].value_counts())

# Define the model with adjusted hyperparameters
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=1, max_iter=1000)

# Train the model
model.fit(train_scaled, train["Target"])

# Predict and evaluate
preds = model.predict(test_scaled)
preds = pd.Series(preds, index=test.index)

# Evaluate precision, recall, and accuracy
precision = precision_score(test["Target"], preds)
recall = recall_score(test["Target"], preds)
accuracy = accuracy_score(test["Target"], preds)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("Accuracy Score:", accuracy)