import ee
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 1. GEE başlatma örnek 'asdasd-12345'
MY_PROJECT_ID = '' 

try:
    ee.Initialize(project=MY_PROJECT_ID)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=MY_PROJECT_ID)

print("Google Earth Engine bağlantısı kuruldu. 42 yıllık Landsat arşivi taranıyor...")
print("Hız optimizasyonu aktif. Pikseller toplanıyor (Lütfen terminali kapatmayın)...")

# 2. Bölge seçimi bismil olarak kullandık
lon, lat = 40.65, 37.85
poi = ee.Geometry.Point([lon, lat])

# bulut maskeleme fonskiyonları
def maskL457(image):
    qa = image.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 5))
    return image.updateMask(cloud.Not())

def maskL89(image):
    qa = image.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4))
    return image.updateMask(cloud.Not())

raw_data = []
zaman_sayaci = 1

current_year = datetime.now().year
current_month = datetime.now().month

# 1984'ten 2026'ya kadar döngü
for year in range(1984, current_year + 1):
    for month in range(1, 13):
        if year == current_year and month > current_month:
            break
            
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
        
        try:
            # Nokta atışı uydu seçimiyle maksimum hız (Merge yok)
            if 1984 <= year <= 1998:
                # Sadece Landsat 5 var (Hızlı)
                collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(poi).filterDate(start_date, end_date).map(maskL457)
                b_green, b_swir = 'SR_B2', 'SR_B5'
            elif 1999 <= year <= 2012:
                # Sadece Landsat 7 (Landsat 5'ten daha temiz veri hattı)
                collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(poi).filterDate(start_date, end_date).map(maskL457)
                b_green, b_swir = 'SR_B2', 'SR_B5'
            elif 2013 <= year <= 2021:
                # Sadece Landsat 8
                collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(poi).filterDate(start_date, end_date).map(maskL89)
                b_green, b_swir = 'SR_B3', 'SR_B6'
            else:
                # Landsat 9 (2022 - 2026)
                collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(poi).filterDate(start_date, end_date).map(maskL89)
                b_green, b_swir = 'SR_B3', 'SR_B6'
        
            if collection.size().getInfo() > 0:
                monthly_image = collection.median()
                ndwi = monthly_image.normalizedDifference([b_green, b_swir]).rename('NDWI')
                
                stats = ndwi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=poi,
                    scale=30
                ).getInfo()
                
                ndwi_value = stats.get('NDWI')
                
                if ndwi_value is not None and not np.isnan(ndwi_value):
                    raw_data.append({
                        'Zaman_Adimi': zaman_sayaci,
                        'Yil': year,
                        'Ay': month,
                        'Tarih': f"{year}-{month:02d}",
                        'NDWI': ndwi_value
                    })
                    zaman_sayaci += 1
        except Exception as e:
            continue

# 3. Verileri dönüştürme ve kaydetme
if raw_data:
    df = pd.DataFrame(raw_data)
    
    # Eksik verileri interpolasyon ile doldur
    df['NDWI'] = df['NDWI'].interpolate(method='linear')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "landsat_42_yillik_nem_verisi.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
    print(f"Toplam Çekilen Ay Sayısı: {len(df)}")
    print(f"Veri Seti Başlangıç Tarihi: {df['Tarih'].min()}")
    print(f"Veri Seti Bitiş Tarihi: {df['Tarih'].max()}")
    print(f"CSV Dosyası Buraya Kaydedildi:\n{csv_path}")
else:
    print("Hata: Belirtilen koordinat veya tarih aralığında uydu verisi bulunamadı.")