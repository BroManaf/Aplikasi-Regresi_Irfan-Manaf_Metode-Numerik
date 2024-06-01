import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Membaca data dari file CSV lokal
file_path = 'Student_Performance (1).csv'  # Ganti dengan path lengkap ke file CSV Anda
data = pd.read_csv(file_path)

# Memilih kolom yang relevan
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
NT = data['Performance Index'].values

# Langkah-langkah Regresi Eksponensial
print("Langkah-langkah Regresi Eksponensial:")
# 1. Mendefinisikan fungsi model eksponensial
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# 2. Menggunakan curve_fit untuk mencari parameter optimal a dan b
popt, pcov = curve_fit(exponential_model, NL.ravel(), NT, p0=(1, 0.01))

# 3. Memprediksi nilai NT berdasarkan model eksponensial yang telah dilatih
NT_pred_exp = exponential_model(NL, *popt)

# 4. Menampilkan parameter dari model eksponensial
print(f"Parameter a: {popt[0]}")
print(f"Parameter b: {popt[1]}")

# Menghitung galat RMS untuk model regresi eksponensial
rms_exp = np.sqrt(mean_squared_error(NT, NT_pred_exp))

# Menampilkan galat RMS
print(f"\nRMS Regresi Eksponensial: {rms_exp}")

# Plot hasil regresi eksponensial
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, NT_pred_exp, color='green', label='Regresi Eksponensial')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Eksponensial')
plt.legend()
plt.show()
