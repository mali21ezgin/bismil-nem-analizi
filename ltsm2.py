import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import os

# 1. CSV verilerinin yüklenmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "landsat_42_yillik_nem_verisi.csv")

if not os.path.exists(csv_path):
    print(f"Hata: '{csv_path}' dosyası bulunamadı!")
    exit()

df = pd.read_csv(csv_path)

# 2. Veri Hazırlığı ve gecikmeli öznitelikler

df['Lag_1'] = df['NDWI'].shift(1)
df['Lag_2'] = df['NDWI'].shift(2)
df['Lag_3'] = df['NDWI'].shift(3)

df_model = df.dropna().copy()

X = df_model[['Zaman_Adimi', 'Ay', 'Lag_1', 'Lag_2', 'Lag_3']].values
y = df_model['NDWI'].values

# Eğitim ve test bölümlenmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. LSTM için veri ölçekleme
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 4. LSTM mimarisi ve eğitimi
lstm_model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

print("LSTM Modeli Eğitiliyor...")
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=4, verbose=1)

y_test_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

# 5. LSTM terminal çıktısı
print("\n==============================================")
print("     --- LSTM MODELİ PERFORMANS METRİKLERİ ---")
print("==============================================")
print(f"MAE  : {mean_absolute_error(y_test, y_test_pred_lstm):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_test_pred_lstm)):.4f}")
print(f"R²   : {r2_score(y_test, y_test_pred_lstm):.4f}")
print("==============================================")

# 6. Gelecek 6 ay ltsm tahminleri
son_satir = df_model.iloc[-1]
son_zaman_adimi = int(son_satir['Zaman_Adimi'])
son_tarih_obj = datetime.strptime(son_satir['Tarih'], "%Y-%m")

pencere_lstm = [son_zaman_adimi, son_tarih_obj.month, son_satir['NDWI'], son_satir['Lag_1'], son_satir['Lag_2']]
tahminler_lstm = []

print("\n--- ÖNÜMÜZDEKİ 6 AYIN LSTM NEM TAHMİNLERİ ---")

for i in range(1, 7):
    gecek_zaman = son_zaman_adimi + i
    gelecek_ay = (son_tarih_obj.month + i - 1) % 12 + 1
    
    girdi_lstm = np.array([pencere_lstm])
    girdi_lstm_scaled = scaler_X.transform(girdi_lstm)
    girdi_lstm_3d = np.reshape(girdi_lstm_scaled, (girdi_lstm_scaled.shape[0], girdi_lstm_scaled.shape[1], 1))
    
    tahmin_lstm = lstm_model.predict(girdi_lstm_3d, verbose=0)[0][0]
    tahminler_lstm.append(tahmin_lstm)
    
    # Terminale 1. ay ile 6. ay arasındaki tahminleri tek tek yazdırma
    print(f"Gelecek {i}. Ay Tahmini NDWI: {tahmin_lstm:.4f}")
    
    pencere_lstm = [gecek_zaman, gelecek_ay, tahmin_lstm, pencere_lstm[2], pencere_lstm[3]]

print("==============================================")

# 7. Sadece LSTM Grafik Çizimi
plt.figure(figsize=(15, 8))
step = max(1, len(df_model) // 25)

# Gerçekleşen Değerler
plt.plot(df_model['Tarih'], df_model['NDWI'], color='blue', alpha=0.4, label='Gerçekleşen Nem (NDWI)', zorder=1)

# LSTM Test Tahmini
plt.plot(df_model['Tarih'].iloc[len(X_train):], y_test_pred_lstm, color='purple', linestyle='--', label='LSTM Test Tahmini')

# Gelecek 6 Ay LSTM Tahmini
gelecek_tarihler = [f"Gelecek {i}.Ay\n(Ay:{(son_tarih_obj.month + i - 1) % 12 + 1 })" for i in range(1, 7)]
plt.plot(gelecek_tarihler, tahminler_lstm, color='darkred', linestyle=':', marker='s', linewidth=2, label='LSTM 6 Ay Tahmini')

plt.title('Bismil Bölgesi Nem Tahmini: LSTM Derin Öğrenme Modeli')
plt.xlabel('Zaman (Seyreltilmiş Görünüm)')
plt.ylabel('Nem Oranı (NDWI)')
plt.xticks(list(df_model['Tarih'])[::step] + gelecek_tarihler, rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(current_dir, "lstm_tahmin_grafigi.png")
plt.savefig(plot_path)
print(f"\nLSTM grafiği kaydedildi:\n{plot_path}")
plt.show()