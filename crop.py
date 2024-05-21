# %%
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from sklearn.decomposition import PCA

# %%
dataset = pd.read_csv("crop_yield_data.csv")

# %%
X = dataset.drop(columns=['yield'])
y = dataset['yield'].values.reshape(-1, 1)

# %%
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# %%
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# %%
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# %%
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# %%
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(X_test, y_test))

# %%
score = model.evaluate(X_test, y_test, batch_size=32)
print("Test loss:", score)

# %%
predictions = model.predict(X_test)

# %%
predictions = scaler_y.inverse_transform(predictions)

# %%
print(predictions)

# %%
y_test = scaler_y.inverse_transform(y_test)

# %%
r2 = r2_score(y_test, predictions)
print("R-squared:", r2)

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs. Predicted Crop Yield")
plt.show()


X = dataset.drop(columns=['Yield','Crop','Crop_Year','State_numeric','Season','Season_numeric','State','Production'])
y = dataset['Production'].values.reshape(-1, 1)