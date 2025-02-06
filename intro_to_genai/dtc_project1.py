import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


df = pd.read_csv('weather_classification_data.csv')

label_encoder = LabelEncoder()


#replacing catoegorical columns with numercial based values 
df['Season'] = label_encoder.fit_transform(df['Season'])
df['Cloud Cover'] = label_encoder.fit_transform(df['Cloud Cover'])
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Weather Type'] = label_encoder.fit_transform(df['Weather Type'])

# choosing X and y
X = df.drop(columns=['Weather Type'])
y = df['Weather Type']


# using train and test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier()

#training the decision tree
dtc.fit(X_train, y_train)

#predicting
y_pred = dtc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Precision and Recall
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for imbalanced datasets
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)