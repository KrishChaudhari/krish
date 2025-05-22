import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
train_df = pd.read_csv(r'C:\Users\hp\Downloads\kaggle_out\train.csv')
test_df = pd.read_csv(r'C:\Users\hp\Downloads\kaggle_out\test.csv')

# Prepare X, y
X = train_df.iloc[:, 1:-1].values
y = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, 2:].values

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Normalize
scaler = MinMaxScaler()
X_res = scaler.fit_transform(X_res)
X_test = scaler.transform(X_test)

# Reshape for LSTM [samples, timesteps, features]
X_res = X_res.reshape((X_res.shape[0], 1, X_res.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Build model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    return model

model = build_model((X_train.shape[1], X_train.shape[2]))

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2, batch_size=256, verbose=1,
    callbacks=callbacks
)

# Predict probabilities
val_probs = model.predict(X_val).flatten()
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = [f1_score(y_val, (val_probs > t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]

# Final predictions
train_preds = (model.predict(X_res) > best_thresh).astype(int).flatten()
y_test_pred = model.predict(X_test)
y_test_pred_labels = (y_test_pred > best_thresh).astype(int).flatten()

# Metrics
print("=== Train Metrics ===")
print(classification_report(y_res, train_preds, digits=4))

if 'label' in test_df.columns:
    y_test_true = test_df['label'].values
    print("\n=== Test Metrics ===")
    print(classification_report(y_test_true, y_test_pred_labels, digits=4))

# Confusion Matrix & Per-Class Accuracy
conf_matrix = confusion_matrix(y_res, train_preds)
tn, fp, fn, tp = conf_matrix.ravel()
class0_accuracy = tn / (tn + fp)
class1_accuracy = tp / (tp + fn)

print("\n=== Confusion Matrix ===")
print(conf_matrix)
print(f"Class 0 Accuracy: {class0_accuracy:.4f}")
print(f"Class 1 Accuracy: {class1_accuracy:.4f}")

if 'label' in test_df.columns:
    y_test_true = test_df['label'].values
    print("\n=== Test Metrics ===")
    print(classification_report(y_test_true, y_test_pred_labels, digits=4))

# Save Submission
submission = pd.DataFrame({
    'Id': test_df.index,
    'Prediction': y_test_pred_labels
})
output_dir = r'C:\Users\hp\Downloads\new submission'
os.makedirs(output_dir, exist_ok=True)
submission_path = os.path.join(output_dir, 'submission2.csv')
submission.to_csv(submission_path, index=False)
print(f"Saved: {submission_path}")