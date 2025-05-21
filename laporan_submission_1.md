# Laporan Proyek Machine Learning - Fadlan Dwi Febrio

## Domain Proyek: Kesehatan

### Latar Belakang

Kesehatan ibu dan bayi merupakan salah satu indikator penting yang mencerminkan kualitas sistem kesehatan suatu negara. Salah satu tantangan terbesar dalam dunia kesehatan ibu dan anak adalah deteksi dini risiko kesehatan janin (fetal health). Risiko ini, jika tidak terdeteksi atau tertangani dengan baik, dapat berujung pada komplikasi serius hingga kematian. 

Cardiotocography (CTG) adalah salah satu alat yang digunakan untuk memantau kesehatan janin. CTG menghasilkan data berupa sinyal detak jantung janin dan kontraksi rahim yang dapat digunakan untuk mendiagnosis potensi masalah kesehatan. Namun, analisis manual terhadap CTG bersifat subyektif, membutuhkan keahlian tinggi, dan rentan terhadap kesalahan manusia.

Pemanfaatan machine learning dalam analisis CTG dapat membantu tenaga medis untuk:
1. Mengotomatisasi proses klasifikasi kondisi janin.
2. Memberikan hasil yang lebih cepat dan akurat.
3. Mengurangi beban kerja tenaga medis, terutama di daerah dengan keterbatasan ahli.

Dengan demikian, proyek ini bertujuan untuk mengembangkan model machine learning yang dapat mengklasifikasikan kondisi kesehatan janin berdasarkan data CTG.

---

## Business Understanding

### Problem Statements

1. Bagaimana cara mengklasifikasikan kondisi kesehatan janin (normal, suspect, patologis) berdasarkan data CTG menggunakan machine learning?
2. Apakah model yang dibangun dapat memberikan hasil yang akurat dan dapat diandalkan untuk digunakan sebagai alat bantu tenaga medis?

### Goals

1. Membangun model machine learning untuk klasifikasi kesehatan janin berdasarkan data CTG.
2. Mengukur performa model menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.
3. Memastikan model dapat diimplementasikan dalam skenario nyata untuk mendukung pengambilan keputusan medis.

### Solution Statements

Untuk mencapai tujuan tersebut, langkah-langkah berikut akan dilakukan:
- **Menggunakan algoritma Neural Network (deep learning)** sebagai baseline model untuk klasifikasi.
- **Melakukan preprocessing data**, termasuk standarisasi fitur dan encoding label target.
- **Melakukan evaluasi performa model** menggunakan metrik klasifikasi (accuracy, precision, recall, F1-score, confusion matrix).
- **Melakukan optimisasi model** dengan hyperparameter tuning untuk meningkatkan performa.

---

## Data Understanding

### Informasi Dataset

Dataset yang digunakan adalah **Fetal Health Classification Dataset**, yang berisi data hasil pemeriksaan CTG. Dataset ini dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

**Dataset Overview**:
- **Jumlah data**: 2,126 sampel
- **Jumlah fitur**: 21 fitur numerik
- **Label target**: `fetal_health` dengan 3 kelas (1=Normal, 2=Suspect, 3=Pathological)

### Variabel-Variabel dalam Dataset

Berikut adalah beberapa variabel penting dalam dataset:
- `baseline value`: Detak jantung dasar janin (bpm).
- `accelerations`: Jumlah akselerasi detak jantung janin per detik.
- `fetal_movement`: Gerakan janin per detik.
- `uterine_contractions`: Kontraksi rahim per detik.
- `light_decelerations`: Deselerasi ringan per detik.
- `severe_decelerations`: Deselerasi berat per detik.
- `abnormal_short_term_variability`: Variasi jangka pendek abnormal (ms).
- `mean_value_of_short_term_variability`: Nilai rata-rata variasi jangka pendek (ms).
- `histogram_mode`: Modus histogram.
- `fetal_health`: Label target (1=Normal, 2=Suspect, 3=Pathological).

### Exploratory Data Analysis (EDA)

1. **Distribusi Kelas Target**:
   Berikut adalah distribusi kelas target dalam dataset:

   ![Distribusi Kelas](https://via.placeholder.com/600x300?text=Distribusi+Kelas+Fetal+Health)

   Dari visualisasi ini, terlihat bahwa kelas `1` (Normal) memiliki jumlah sampel yang jauh lebih banyak dibandingkan kelas `2` (Suspect) dan `3` (Pathological). Ketidakseimbangan kelas ini dapat memengaruhi performa model, terutama pada recall kelas minoritas.

2. **Heatmap Korelasi Antar Fitur**:
   Berikut adalah heatmap korelasi antar fitur dalam dataset:

   ![Heatmap Korelasi](https://via.placeholder.com/600x300?text=Heatmap+Korelasi)

   Beberapa fitur menunjukkan korelasi tinggi, seperti `baseline value` dan `histogram_mean`. Hal ini dapat memengaruhi performa model.

---

## Data Preparation

Berikut adalah langkah-langkah yang dilakukan untuk mempersiapkan data:

1. **Membaca Data**:
   ```python
   df = pd.read_csv("fetal_health.csv")
   ```

2. **Mengecek Missing Value**:
   Tidak ditemukan missing value dalam dataset.

3. **Membagi Fitur dan Target**:
   ```python
   X = df.drop('fetal_health', axis=1)
   y = df['fetal_health']
   ```

4. **Membagi Data Train-Test**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   ```

5. **Standarisasi Data**:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

6. **One-Hot Encoding untuk Target**:
   ```python
   from tensorflow.keras.utils import to_categorical
   y_train_encoded = to_categorical(y_train - 1)
   y_test_encoded = to_categorical(y_test - 1)
   ```

---

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

   ![Learning Curve](https://via.placeholder.com/600x300?text=Learning+Curve)

2. **Confusion Matrix**:
   Berikut adalah confusion matrix dari hasil prediksi model pada data test:

   ![Confusion Matrix](https://via.placeholder.com/600x300?text=Confusion+Matrix)

3. **Classification Report**:
   Berikut adalah classification report yang menunjukkan metrik evaluasi per kelas:
   ```
   precision    recall   f1-score   support
   0 (Normal)        0.93      0.98      0.95       332
   1 (Suspect)       0.76      0.58      0.65        59
   2 (Pathological)  0.81      0.74      0.78        35

   accuracy                               0.90       426
   macro avg          0.83      0.77      0.79       426
   weighted avg       0.89      0.90      0.90       426
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
