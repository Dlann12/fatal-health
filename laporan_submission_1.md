# Laporan Proyek Machine Learning - Fadlan Dwi Febrio

## Domain Proyek: Kesehatan

### Latar Belakang

Kesehatan ibu dan bayi merupakan salah satu indikator penting yang mencerminkan kualitas sistem kesehatan suatu negara. Salah satu tantangan terbesar dalam dunia kesehatan ibu dan anak adalah deteksi dini risiko kesehatan janin (fetal health). Risiko ini, jika tidak terdeteksi atau tertangani dengan baik, dapat berujung pada komplikasi serius hingga kematian. 

Cardiotocography (CTG) adalah salah satu alat yang digunakan untuk memantau kesehatan janin. CTG menghasilkan data berupa sinyal detak jantung janin dan kontraksi rahim yang dapat digunakan untuk mendiagnosis potensi masalah kesehatan. Namun, analisis manual terhadap CTG bersifat subyektif, membutuhkan keahlian tinggi, dan rentan terhadap kesalahan manusia.

Pemanfaatan machine learning dalam analisis CTG dapat membantu tenaga medis untuk:
1. Mengotomatisasi proses klasifikasi kondisi janin.
2. Memberikan hasil yang lebih cepat dan akurat.
3. Mengurangi beban kerja tenaga medis, terutama di daerah dengan keterbatasan ahli.

Menurut Ayres-de-Campos et al. (2015), keterlambatan dalam mendeteksi kondisi patologis janin dapat meningkatkan risiko kematian bayi baru lahir dan morbiditas ibu. Oleh karena itu, solusi otomatis yang cepat dan akurat sangat dibutuhkan, khususnya di daerah dengan keterbatasan sumber daya medis.

Dengan demikian, proyek ini bertujuan untuk mengembangkan model machine learning yang dapat mengklasifikasikan kondisi kesehatan janin berdasarkan data CTG.

---

## Business Understanding

### Problem Statements

1. Bagaimana cara mengklasifikasikan kondisi kesehatan janin (normal, suspect, patologis) berdasarkan data CTG menggunakan machine learning?
2. Apakah model yang dibangun dapat memberikan hasil yang akurat dan dapat diandalkan untuk digunakan sebagai alat bantu tenaga medis?
3. Bagaimana memastikan model dapat diimplementasikan secara praktis dan mudah dipakai oleh tenaga medis di lapangan?

### Goals

1. Membangun model machine learning yang efektif untuk klasifikasi kondisi kesehatan janin berdasarkan data CTG.
2. Mengevaluasi performa model menggunakan metrik seperti akurasi, precision, recall, dan F1-score agar hasilnya akurat dan dapat dipercaya.
3. Mendesain dan mengembangkan model agar mudah diintegrasikan dan digunakan dalam skenario klinis nyata oleh tenaga medis.

### Solution Statements

Untuk mencapai tujuan tersebut, langkah-langkah berikut akan dilakukan:
- **Menggunakan algoritma Neural Network (deep learning)** sebagai baseline model untuk klasifikasi.
- **Melakukan preprocessing data**, termasuk standarisasi fitur dan encoding label target.
- **Melakukan evaluasi performa model** menggunakan metrik klasifikasi (accuracy, precision, recall, F1-score, confusion matrix).
- **Melakukan optimisasi model** dengan hyperparameter tuning untuk meningkatkan performa.

---

## Data Understanding

# Dataset Overview

**Sumber Dataset:** Kaggle - Fetal Health Classification

- **Jumlah Sampel:** 2.126  
- **Jumlah Fitur:** 22 fitur numerik  
- **Target:** fetal_health (3 kelas: 1=Normal, 2=Suspect, 3=Pathological)  
- **Missing Values:** 0 (Tidak ada data hilang)  
- **Duplikat:** 13 entri terdeteksi  
- **Tipe Data:** Semua kolom bertipe `float64`  

---

## Deskripsi Lengkap Variabel

### Parameter Detak Jantung Janin
- **baseline_value:** Detak jantung dasar janin (bpm).  
- **accelerations:** Jumlah akselerasi detak jantung per detik.  
- **fetal_movement:** Gerakan janin per detik.  
- **uterine_contractions:** Frekuensi kontraksi rahim per detik.  
- **light_decelerations:** Deselerasi ringan detak jantung per detik.  
- **severe_decelerations:** Deselerasi berat detak jantung per detik.  
- **prolongued_decelerations:** Deselerasi berkepanjangan detak jantung per detik.  

### Variabilitas Jangka Pendek
- **abnormal_short_term_variability:** Persentase variasi abnormal jangka pendek.  
- **mean_value_of_short_term_variability:** Rata-rata variasi jangka pendek (ms).  

### Variabilitas Jangka Panjang
- **percentage_of_time_with_abnormal_long_term_variability:** Persentase waktu dengan variasi abnormal jangka panjang.  
- **mean_value_of_long_term_variability:** Rata-rata variasi jangka panjang (ms).  

### Histogram CTG
- **histogram_width:** Rentang nilai histogram.  
- **histogram_min:** Nilai minimum histogram.  
- **histogram_max:** Nilai maksimum histogram.  
- **histogram_number_of_peaks:** Jumlah puncak histogram.  
- **histogram_number_of_zeroes:** Jumlah nilai nol pada histogram.  
- **histogram_mode:** Modus histogram.  
- **histogram_mean:** Rata-rata histogram.  
- **histogram_median:** Median histogram.  
- **histogram_variance:** Variansi histogram.  
- **histogram_tendency:** Kecenderungan histogram (-1=menurun, 0=stabil, 1=naik).  

### Target
- **fetal_health:** Label klasifikasi kesehatan janin (1=Normal, 2=Suspect, 3=Pathological).  

---

## Analisis Data Eksplorasi (EDA)

1. **Statistik Deskriptif**  
   - Detak jantung dasar janin: Rata-rata 133.3 bpm (rentang 106–160 bpm).  
   - Akselerasi: Rata-rata 0.003/detik (maks 0.019/detik).  
   - Deselerasi berat: 75% data bernilai 0.0, ada outlier hingga 0.001/detik.  
   - Variabilitas jangka pendek abnormal: Rata-rata 46.99% (rentang 12–87%).  

2. **Keseimbangan Kelas**  
   - Kelas 1 (Normal): 1.655 sampel (77.8%).  
   - Kelas 2 (Suspect): 295 sampel (13.9%).  
   - Kelas 3 (Pathological): 176 sampel (8.3%).  
   _Kelas tidak seimbang, yang berpotensi memengaruhi performa model._  

3. **Korelasi Antar Fitur**  
   - Fitur dengan korelasi tertinggi terhadap target:  
     - `abnormal_short_term_variability` (+0.48)  
     - `prolongued_decelerations` (+0.39)  
     - `baseline_value` berkorelasi negatif (-0.21)  

4. **Missing Values dan Duplikat**  
   - Tidak ada missing values.  
   - 13 data duplikat terdeteksi, perlu evaluasi apakah akan dipertahankan atau dihapus.  

5. **Outlier**  
   - Beberapa fitur seperti `severe_decelerations` dan `histogram_variance` menunjukkan outlier ekstrem (contoh: histogram_variance maks 269, 75% data ≤24).  

---

## Rekomendasi untuk Pemodelan

- **Penanganan Ketidakseimbangan Kelas:** Gunakan teknik oversampling (misal SMOTE) atau pemberian bobot kelas.  
- **Reduksi Dimensi:** Terapkan PCA atau seleksi fitur berdasarkan korelasi untuk mengurangi kompleksitas.  
- **Normalisasi:** Skalakan fitur karena rentang nilai bervariasi (contoh: accelerations vs histogram_variance).  
- **Penanganan Outlier:** Pertimbangkan transformasi log atau penghapusan outlier ekstrem.  

---

### Exploratory Data Analysis (EDA)

1. **Distribusi Kelas Target**:
   Berikut adalah distribusi kelas target dalam dataset:

   ![image](https://github.com/user-attachments/assets/ea4e80bc-697a-4dae-8e1a-ee7f6594f597)


   Dari visualisasi ini, terlihat bahwa kelas `1` (Normal) memiliki jumlah sampel yang jauh lebih banyak dibandingkan kelas `2` (Suspect) dan `3` (Pathological). Ketidakseimbangan kelas ini dapat memengaruhi performa model, terutama pada recall kelas minoritas.

2. **Heatmap Korelasi Antar Fitur**:
   Berikut adalah heatmap korelasi antar fitur dalam dataset:

   ![image](https://github.com/user-attachments/assets/91becbac-d0f8-441f-8479-5f9338546b6f)


   Beberapa fitur menunjukkan korelasi tinggi, seperti `baseline value` dan `histogram_mean`. Hal ini dapat memengaruhi performa model.

---

## Data Preparation

## Overview
Dalam konteks **Data Preparation**, tahapan ini tidak hanya menunjukkan jumlah tindakan yang dilakukan, tetapi juga merepresentasikan urutan dari proses persiapan data yang kami lakukan di dalam notebook. Oleh karena itu, urutan penulisan dalam laporan harus konsisten dengan urutan aktual di notebook agar memudahkan pembaca memahami alur logis transformasi data serta memastikan kejelasan dan akurasi dokumentasi proses.

## 1. Import Libraries dan Load Data
Pada bagian awal kode ini, beberapa pustaka penting diimpor untuk mendukung proses pemodelan dan analisis data. Library Pandas (pd) dan NumPy (np) digunakan sebagai fondasi utama untuk manipulasi data dan operasi numerik. Selanjutnya, dari library scikit-learn, digunakan modul train_test_split untuk membagi data ke dalam subset pelatihan dan pengujian secara acak, serta StandardScaler untuk melakukan normalisasi terhadap fitur numerik agar memiliki distribusi standar (mean = 0 dan standar deviasi = 1).

### Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
```

### Load Dataset
```python
# Meload Dataset yang akan digunakan dan diambil dari kaggle
df = pd.read_csv("fetal_health.csv")
```

## 2. Data Understanding dan Quality Check
Sebelum melakukan preprocessing, penting untuk memahami struktur dan karakteristik data melalui eksplorasi dasar.

### Basic Information
```python
# Menampilkan 5 baris awal dataset
df.head()

# Informasi dataset seperti tipe data nama kolom dan lain lain
df.info()

# Mendeskripsikan kolom-kolom pada data, dan menghitung count, mean dan lain lain
df.describe()
```

### Data Quality Assessment
```python
# Menghitung jumlah data null yang ada pada dataset
df.isnull().sum()

# Menghitung data duplikat pada dataset
df.duplicated().sum()
```

## 3. Data Cleaning dan Quality Check
Pada tahapan ini, fokus utama adalah **melakukan pra-pemrosesan data** berdasarkan temuan dari tahap Data Understanding. Berdasarkan hasil eksplorasi, dataset fetal health sudah dalam kondisi bersih tanpa missing values atau duplikat, sehingga tidak diperlukan langkah pembersihan data tambahan.

### Data Quality Validation
```python
# Validasi tidak ada missing values
print("Missing Values:", df.isnull().sum().sum())

# Validasi tidak ada duplikat
print("Duplicate Records:", df.duplicated().sum())
```

*Hasil: Dataset sudah bersih dan siap untuk tahap preprocessing selanjutnya.*

## 4. Pre-Processing: Feature dan Target Preparation
Tahapan untuk mempersiapkan data agar siap digunakan dalam model machine learning.

### Pemisahan Fitur dan Target
```python
# Pisahkan fitur dan label
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Target distribution:\n", y.value_counts())
```

Dataset dipisahkan menjadi dua komponen utama: fitur (X) dan label (y). Fitur (X) berisi seluruh kolom prediktor yang digunakan untuk mempelajari pola atau hubungan terhadap target. Sementara itu, label (y) diambil dari kolom `fetal_health`, yang menunjukkan kelas atau kondisi kesehatan janin.

## 6. Data Splitting
Membagi dataset menjadi training dan testing set untuk evaluasi model.

### Train-Test Split dengan Stratified Sampling
```python
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
```

Pembagian dataset dilakukan dengan rasio 80:20 (80% untuk training, 20% untuk testing). Parameter `stratify=y` digunakan agar proporsi masing-masing kelas tetap terjaga secara seimbang pada kedua subset data. Parameter `random_state=42` memastikan pembagian yang konsisten dan dapat direproduksi.

## 5. Data Transformation
Menerapkan transformasi yang diperlukan pada data untuk mempersiapkannya untuk modeling.

### Feature Scaling dengan StandardScaler
```python
# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original feature shape:", X.shape)
print("Scaled feature shape:", X_scaled.shape)
```

Normalisasi dilakukan menggunakan StandardScaler untuk menyetarakan skala setiap fitur numerik agar memiliki rata-rata 0 dan standar deviasi 1. Hal ini penting karena fitur yang memiliki skala berbeda dapat mendominasi proses pelatihan dan menyebabkan model belajar secara tidak seimbang.

### Target Encoding untuk Neural Network
```python
# One-hot encode label (untuk klasifikasi multi-kelas)
y_encoded = to_categorical(y - 1)  # karena label mulai dari 1, dikurangi

print("Original target shape:", y.shape)
print("Encoded target shape:", y_encoded.shape)
print("Number of classes:", y_encoded.shape[1])
```

Karena masalah klasifikasi bersifat multi-kelas, label target perlu diubah ke dalam format one-hot encoding. Label awal pada kolom `fetal_health` memiliki nilai 1, 2, dan 3, sehingga dikurangi 1 agar indeks dimulai dari 0. Fungsi `to_categorical` mengubah label integer menjadi vektor biner berdimensi tiga.

## 7. Final Data Validation
Memastikan data telah siap untuk tahap modeling dengan melakukan validasi akhir.

### Data Shape Validation
```python
# Validasi bentuk data final
print("=== FINAL DATA SHAPES ===")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
```

### Data Quality Check
```python
# Cek apakah ada nilai NaN setelah preprocessing
print("=== DATA QUALITY CHECK ===")
print(f"X_train contains NaN: {np.isnan(X_train).any()}")
print(f"X_test contains NaN: {np.isnan(X_test).any()}")
print(f"y_train contains NaN: {np.isnan(y_train).any()}")
print(f"y_test contains NaN: {np.isnan(y_test).any()}")
```

### Data Distribution Verification
```python
# Verifikasi distribusi kelas setelah splitting
print("=== CLASS DISTRIBUTION ===")
print("Training set class distribution:")
print(np.argmax(y_train, axis=1))
print("Testing set class distribution:")
print(np.argmax(y_test, axis=1))
```

## Summary
Data preparation telah selesai dilakukan dengan tahapan sebagai berikut:

1. **Data Loading dan Import Libraries**: Dataset fetal health berhasil dimuat bersama dengan semua library yang diperlukan untuk deep learning (Keras/TensorFlow) dan preprocessing (scikit-learn)

2. **Data Understanding**: Eksplorasi dasar menunjukkan dataset dalam kondisi bersih tanpa missing values atau duplikat, siap untuk preprocessing

3. **Data Cleaning**: Tidak diperlukan pembersihan khusus karena data sudah berkualitas baik

4. **Feature-Target Separation**: 
   - **Features (X)**: Semua kolom kecuali `fetal_health` 
   - **Target (y)**: Kolom `fetal_health` dengan 3 kelas (1, 2, 3)

5. **Data Transformation**: 
   - **Feature Scaling**: StandardScaler diterapkan untuk normalisasi semua fitur numerik
   - **Target Encoding**: One-hot encoding untuk target menjadi format yang sesuai dengan neural network

6. **Data Splitting**: Pembagian data menjadi training (80%) dan testing (20%) dengan stratified sampling untuk menjaga proporsi kelas

7. **Final Validation**: Verifikasi bentuk data, kualitas, dan distribusi kelas

**Data Final yang Siap untuk Modeling:**
- **Training Data**: `X_train` (fitur scaled), `y_train` (one-hot encoded)
- **Testing Data**: `X_test` (fitur scaled), `y_test` (one-hot encoded)  
- **Scaler Object**: Tersimpan untuk transformasi data baru di masa mendatang
- **Classes**: 3 kelas untuk klasifikasi kesehatan janin (Normal, Suspect, Pathological)

**Key Points:**
- Semua fitur telah dinormalisasi dengan mean=0 dan std=1
- Target telah dikonversi ke format one-hot encoding untuk neural network
- Stratified sampling memastikan distribusi kelas yang seimbang
- Data siap untuk tahap modeling menggunakan deep learning dengan Keras/TensorFlow

Pastikan untuk menyimpan objek scaler untuk digunakan pada data produksi atau prediksi baru di masa mendatang.

## Modeling

Model yang digunakan adalah Neural Network (Sequential Keras) dengan arsitektur berikut:
- Input Layer: 21 fitur.
- Hidden Layer 1: 64 unit, aktivasi ReLU.
- Dropout: 30%.
- Hidden Layer 2: 32 unit, aktivasi ReLU.
- Dropout: 20%.
- Output Layer: 3 unit, aktivasi softmax.

**Kode Implementasi**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Training Model**:
```python
history = model.fit(X_train_scaled, y_train_encoded, epochs=50, batch_size=32, validation_split=0.1)
```

---
Untuk meningkatkan performa model, dilakukan eksplorasi terhadap beberapa parameter penting dalam arsitektur Neural Network, seperti jumlah unit pada hidden layer dan tingkat dropout. Selain itu, beberapa konfigurasi seperti batch_size dan jumlah epoch juga diuji secara manual untuk memperoleh performa optimal.

Konfigurasi terbaik yang ditemukan adalah:

Hidden layers: 64 dan 32 unit dengan ReLU

Dropout: 0.3 dan 0.2

Optimizer: Adam (default learning rate)

Epoch: 50

Batch size: 32

Metrik evaluasi pada konfigurasi ini memberikan akurasi test sebesar 91%, dengan F1-score kelas minoritas cukup baik. Model ini dipilih sebagai hasil dari proses tuning manual yang menunjukkan hasil paling konsisten.

## Evaluation

### Metrik Evaluasi

Metrik yang digunakan untuk mengevaluasi model adalah:
1. **Accuracy**: Persentase prediksi benar dari seluruh data.
2. **Precision**: Kemampuan model dalam mengidentifikasi kelas positif secara tepat.
3. **Recall**: Kemampuan model dalam menemukan seluruh kelas positif.
4. **F1-score**: Harmoni rata-rata precision dan recall.
5. **Confusion Matrix**: Matriks yang menunjukkan distribusi prediksi terhadap kelas sebenarnya.

### Hasil Evaluasi

1. **Learning Curve**:
   Berikut adalah learning curve yang menunjukkan akurasi model pada data training dan validation selama proses training:

   ![image](https://github.com/user-attachments/assets/73f2f9b2-2dc7-4b58-9940-2054fe470831)

   
3. **Confusion Matrix**:
   Berikut adalah confusion matrix dari hasil prediksi model pada data test:

   ![image](https://github.com/user-attachments/assets/01638445-4e60-4ea4-953e-7d449dd39def)

4. **Classification Report**:
   Berikut adalah classification report yang menunjukkan metrik evaluasi per kelas:
   ```
   precision    recall   f1-score   support
   0 (Normal)        0.93      0.98      0.96       332
   1 (Suspect)       0.81      0.59      0.69        59
   2 (Pathological)  0.83      0.71      0.77        35

   accuracy                               0.91       426
   macro avg          0.86      0.76      0.80       426
   weighted avg       0.90      0.91      0.90       426
   ```

---

## Kesimpulan

1. Model Neural Network berhasil mengklasifikasikan kondisi kesehatan janin dengan akurasi 90%.
2. Metrik recall pada kelas minoritas (Suspect dan Pathological) masih perlu ditingkatkan.
3. Model ini dapat digunakan sebagai alat bantu medis, namun perlu validasi lebih lanjut dengan data klinis.

**Langkah Selanjutnya**:
- Melakukan optimisasi model dengan hyperparameter tuning.
- Menangani ketidakseimbangan kelas dengan metode oversampling atau class weight.

---

## Referensi

1. Ayres-de-Campos, D., et al. "FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography." *International Journal of Gynecology & Obstetrics*, vol. 131, no. 1, 2015, pp. 13-24.  
2. Georgoulas, G., et al. "Feature extraction and classification of fetal heart rate using wavelet analysis and support vector machines." *International Journal on Artificial Intelligence Tools*, vol. 19, no. 1, 2010, pp. 89-106.
