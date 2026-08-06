import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import datetime
import os

# 1. hazır olan csv verisinin yüklenmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "landsat_42_yillik_nem_verisi.csv")

if not os.path.exists(csv_path):
    print(f"Hata: '{csv_path}' dosyası bulunamadı! Lütfen CSV dosyasının kodla aynı klasörde olduğundan emin olun.")
    exit()

print(f"Veriler '{csv_path}' üzerinden yükleniyor...")
df = pd.read_csv(csv_path)

# 2. KAYARAK PENCERE (SLIDING WINDOW) & GEÇMİŞ ÖZNİTELİK EKLEME ---
# Son 3 ayın nem değerlerini (Lag_1, Lag_2, Lag_3) yeni özellik olarak ekliyoruz
df['Lag_1'] = df['NDWI'].shift(1)
df['Lag_2'] = df['NDWI'].shift(2)
df['Lag_3'] = df['NDWI'].shift(3)

# İlk 3 satırda geçmiş veri oluşamayacağı (NaN olacağı) için bu satırları siliyoruz
df_model = df.dropna().copy()

# x ve y leri belirleme
X = df_model[['Zaman_Adimi', 'Ay', 'Lag_1', 'Lag_2', 'Lag_3']].values
y = df_model['NDWI'].values

# 3. Veriyi eğitim ve test olarak bölümleme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Model Eğitiliyor... (Eğitim seti: {len(X_train)} ay, Test seti: {len(X_test)} ay)")

# 4. rf modeli 
model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

print("\n--- 42 YILLIK MODEL TEST SONUÇLARI ---")
y_test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print(f"Test Verisi Ortalama Mutlak Hata (MAE): {mae:.4f}")
print(f"Test Verisi Hata Kareler Ortalamasının Kökü (RMSE): {rmse:.4f}")
print(f"Model Başarı Skoru (R² Skoru): {r2:.4f} (1.0'a ne kadar yakınsa o kadar iyi)")

# 5. Gelecek Tahmini Yapma (Önümüzdeki 6 Ay)
# Gelecek tahminleri zincirleme olacağı için her tahminde bir önceki tahminleri girdi olarak kullanacağız
son_satir = df_model.iloc[-1]
son_zaman_adimi = int(son_satir['Zaman_Adimi'])
son_tarih_obj = datetime.strptime(son_satir['Tarih'], "%Y-%m")

tahminler = []
gecmis_pencere = [son_satir['NDWI'], son_satir['Lag_1'], son_satir['Lag_2']] # Son 3 değerle başlıyoruz

print("\n--- ÖNÜMÜZDEKİ 6 AYIN GERÇEKÇİ NEM TAHMİNLERİ ---")
for i in range(1, 7):
    gecek_zaman = son_zaman_adimi + i
    gelecek_ay = (son_tarih_obj.month + i - 1) % 12 + 1
    
    # Modele girdi hazırlığı: [Zaman_Adimi, Ay, Lag_1, Lag_2, Lag_3]
    girdi_ozellikleri = np.array([[gecek_zaman, gelecek_ay, gecmis_pencere[0], gecmis_pencere[1], gecmis_pencere[2]]])
    
    # Tahmin üretme
    mevcut_tahmin = model.predict(girdi_ozellikleri)[0]
    tahminler.append(mevcut_tahmin)
    print(f"Gelecek {i}. Ay Tahmini NDWI: {mevcut_tahmin:.4f}")
    
    # Pencereyi bir adım kaydır: Yeni tahmini başa ekle, en eskiyi at
    gecmis_pencere = [mevcut_tahmin] + gecmis_pencere[:-1]

# 6. Grafik Çizdirme ve Kaydetme
plt.figure(figsize=(15, 7))

# Grafikteki görsel yoğunluğu azaltmak için x eksenindeki tarih etiketlerini seyreltiyoruz
step = max(1, len(df_model) // 25)

# Gerçekleşen geçmiş veriler
plt.plot(df_model['Tarih'], df_model['NDWI'], color='blue', alpha=0.6, label='Gerçekleşen Nem (NDWI)', zorder=2)

# Modelin eğitim verisindeki başarısı
plt.plot(df_model['Tarih'].iloc[:len(X_train)], model.predict(X_train), color='green', linestyle='--', alpha=0.7, label='Modelin Eğitimi (Train Fit)')

# Modelin test verisindeki başarısı
plt.plot(df_model['Tarih'].iloc[len(X_train):], y_test_pred, color='orange', linestyle='-.', linewidth=2, label='Modelin Test Tahmini (Test Predict)')

# Gelecek Tahminleri Tarih Etiketleri Oluşturma
gelecek_tarihler = []
for i in range(1, 7):
    m = (son_tarih_obj.month + i - 1) % 12 + 1
    gelecek_tarihler.append(f"Gelecek {i}.Ay\n(Ay:{m})")

plt.scatter(gelecek_tarihler, tahminler, color='red', s=100, zorder=5)
plt.plot(gelecek_tarihler, tahminler, color='red', linestyle=':', linewidth=2, label='Gelecek 6 Ay Tahmini')

plt.title('Bismil Bölgesi 42 Yıllık Landsat Verisi ve Gecikmeli Öznitelikler (Lag Features) ile Nem Tahmini')
plt.xlabel('Zaman (Seyreltilmiş Görünüm)')
plt.ylabel('Nem Oranı (NDWI)')

# X ekseni etiketlerini seyreltilmiş tarihler + gelecek tarihler şeklinde birleştirme
plt.xticks(list(df_model['Tarih'])[::step] + gelecek_tarihler, rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(current_dir, "42_yillik_nem_tahmin_grafigi.png")
plt.savefig(plot_path)

print(f"\n42 yıllık analizi içeren yeni grafik başarıyla kaydedildi:\n{plot_path}")
plt.show()