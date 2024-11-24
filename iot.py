import pandas as pd

#  مسیر فایل‌
file_paths = [
    "Thursday-WorkingHours.csv",
    "Tuesday-WorkingHours.csv",
    "Wednesday-workingHours.csv",
    "Friday-WorkingHours.csv",
    "Monday-WorkingHours.csv"
]

# خواندن فایل‌ها و نمایش پیش‌نمایش
for file in file_paths:
    print(f"Reading {file}")
    df = pd.read_csv(file)
    print(df.head())
    print(df.info())
    print("\n")
from sklearn.preprocessing import StandardScaler

# ترکیب فایل‌ها در یک DataFrame واحد
all_data = pd.DataFrame()
for file in file_paths:
    df = pd.read_csv(file)
    all_data = pd.concat([all_data, df])

# حذف ستون‌های غیرضروری
all_data = all_data.drop(columns=['irrelevant_column_1', 'irrelevant_column_2'], errors='ignore')

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
features = scaler.fit_transform(all_data.drop(columns=['Label']))  # 'Label' باید شامل حمله/عدم حمله باشد
labels = all_data['Label']
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# تعریف مدل
model = Sequential([
    LSTM(128, input_shape=(100, features.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # خروجی باینری
])

# کامپایل مدل
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
import numpy as np

# تبدیل داده‌ها به توالی
def create_sequences(data, labels, seq_length=100):
    sequences, sequence_labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        sequence_labels.append(labels[i + seq_length])
    return np.array(sequences), np.array(sequence_labels)

X, y = create_sequences(features, labels)

# تقسیم به داده‌های آموزش و تست
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
from sklearn.metrics import classification_report, roc_auc_score

# پیش‌بینی
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# گزارش ارزیابی
print(classification_report(y_test, y_pred_classes))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))


# مقایسه پیش‌بینی و برچسب واقعی
misclassified_indices = np.where(y_pred_classes.flatten() != y_test.flatten())[0]

# نمایش نمونه‌های اشتباه
misclassified_samples = X_test[misclassified_indices]
actual_labels = y_test[misclassified_indices]
predicted_labels = y_pred_classes[misclassified_indices]

# ایجاد یک DataFrame برای تحلیل
df_errors = pd.DataFrame({
    "Actual Label": actual_labels,
    "Predicted Label": predicted_labels.flatten()
})

print("Sample Errors:")
print(df_errors.head())
# بررسی ویژگی‌های نمونه‌های اشتباه
for i in range(len(misclassified_samples)):
    print(f"Sample {i}:")
    print("Features:", misclassified_samples[i])
    print("Actual Label:", actual_labels[i])
    print("Predicted Label:", predicted_labels[i])
    print("\n")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ایجاد ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Attack"])
disp.plot()
# بررسی کلاس‌های اشتباه
print("Errors by Class:")
print(df_errors.groupby("Actual Label")["Predicted Label"].value_counts())
import matplotlib.pyplot as plt

# نمایش پراکندگی ویژگی‌ها
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.6, label="Actual")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_classes.flatten(), cmap="viridis", alpha=0.3, label="Predicted")
plt.legend()
plt.show()


