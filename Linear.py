import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Membaca dataset dari file lokal
file_path = 'Student_Performance (1).csv'  # Gantilah path ini dengan lokasi sebenarnya dari file dataset Anda
data = pd.read_csv(file_path)

# Mengambil kolom yang relevan
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
NT = data['Performance Index'].values

# Membagi data menjadi training dan testing set
NL_train, NL_test, NT_train, NT_test = train_test_split(NL, NT, test_size=0.2, random_state=42)

# Membuat model regresi linier
model = LinearRegression()
model.fit(NL_train, NT_train)

# Prediksi
NT_pred_train = model.predict(NL_train)
NT_pred_test = model.predict(NL_test)

# Menghitung RMS error
rms_error_train = np.sqrt(mean_squared_error(NT_train, NT_pred_train))
rms_error_test = np.sqrt(mean_squared_error(NT_test, NT_pred_test))

# Plot grafik titik data dan hasil regresi
plt.figure(figsize=(10, 6))
plt.scatter(NL, NT, color='blue', label='Data Sebenarnya')
plt.plot(NL, model.predict(NL), color='red', label='Garis Regresi')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linier Jumlah Latihan Soal terhadap Nilai Ujian')
plt.legend()
plt.show()

# Menampilkan hasil RMS error
print(f'RMS Error (Training set): {rms_error_train}')
print(f'RMS Error (Testing set): {rms_error_test}')
