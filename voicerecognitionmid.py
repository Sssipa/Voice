import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Akuisisi Data/Pengumpulan Data
def load_dataset(dataset_path):
    features = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if file_path.endswith('.wav'):
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(label)
    return np.array(features), np.array(labels)

# Pre-Processing
def pre_process_data(X):
    X = np.nan_to_num(X)  # Menangani nilai NaN atau tak terdefinisi
    return X

# Ekstraksi Fitur
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Ekstraksi MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc.flatten()  # Meratakan MFCC menjadi satu vektor
    return mfcc

# Pembuatan Model/Algoritma
models = {
    'K-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Path dataset
dataset_path = 'E:\Coding\Semester 3\Pengenalan Pola\Voice\dataset'  # Ganti dengan path ke folder dataset Anda

# Pelatihan Model/Algoritma
X, y = load_dataset(dataset_path)

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Pre-Processing
X = pre_process_data(X)

# Membagi data menjadi training dan testing (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Fungsi untuk menampilkan hasil metrik dan confusion matrix tiap model
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    # Menampilkan metrik untuk tiap model
    print(f"\nAkurasi untuk {model_name}: {accuracy:.2f}%")
    print(f"Precision untuk {model_name}: {precision:.2f}%")
    print(f"Recall untuk {model_name}: {recall:.2f}%")
    print(f"F1-Score untuk {model_name}: {f1:.2f}%")
    
    # Menampilkan confusion matrix untuk tiap model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix untuk {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Melatih dan menguji tiap model
for model_name, model in models.items():
    evaluate_model(model, X_train, y_train, X_test, y_test, model_name)

# Visualisasi Kinerja Model
accuracies = []
model_names = ['K-NN', 'Naive Bayes', 'SVM', 'Random Forest']
precision_list = []
recall_list = []
f1_list = []

for model_name, model in models.items():
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    
    accuracies.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

plt.figure(figsize=(8, 6))
plt.bar(model_names, accuracies, color='skyblue', label='Accuracy')
plt.bar(model_names, precision_list, color='orange', label='Precision')
plt.bar(model_names, recall_list, color='green', label='Recall')
plt.bar(model_names, f1_list, color='red', label='F1-Score')

plt.title('Perbandingan Akurasi, Precision, Recall, F1-Score Model Voice Recognition')
plt.xlabel('Model')
plt.ylabel('Score (%)')
plt.ylim(0, 100)
plt.legend()
plt.show()

