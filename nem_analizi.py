import ee
import geemap

#google cloud bilgileri
# Örnek: 'asdasd' veya 'asdasd-123456' şeklinde olmalı.
MY_PROJECT_ID = '' 

# 1. Earth Engine Doğrulaması ve başlatılması
try:
    print(f"Earth Engine '{MY_PROJECT_ID}' projesi ile başlatılıyor...")
    ee.Initialize(project=MY_PROJECT_ID)
except Exception as e:
    print("Giriş başarısız oldu, yeniden doğrulama başlatılıyor...")
    ee.Authenticate()
    ee.Initialize(project=MY_PROJECT_ID)

print("Google Earth Engine başarıyla başlatıldı!")

# 2. bölgenin koordinatları
lon, lat = 40.65, 37.85 
poi = ee.Geometry.Point([lon, lat])

# 3. Sentinel-2 filtrelemesi
start_date = '2026-05-01'
end_date = '2026-06-30'

dataset = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(poi)
           .filterDate(start_date, end_date)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) 
           .sort('CLOUDY_PIXEL_PERCENTAGE')
)

image = dataset.first()

# Görüntünün hangi tarihe ait olduğunu konsola yazdıralım
date = image.date().format('YYYY-MM-DD').getInfo()
print(f"Analiz edilen en temiz uydu görüntüsünün tarihi: {date}")

# 4. NDWI (Nem İndeksi) Hesaplama
ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

# 5. ısı haritası
Map = geemap.Map(center=[lat, lon], zoom=12)

rgb_vis = {
    'min': 0.0,
    'max': 3000.0,
    'bands': ['B4', 'B3', 'B2'],
}
Map.addLayer(image, rgb_vis, 'Gerçek Renkli Görüntü (RGB)')

ndwi_vis = {
    'min': -0.5,
    'max': 0.5,
    'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#053061']
}
Map.addLayer(ndwi, ndwi_vis, 'Toprak/Bitki Nem Haritası (NDWI)')
Map.addLayerControl()

# Haritayı doğrudan klasörün içine kaydetme
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_html = os.path.join(current_dir, "bismil_nem_analizi.html")

Map.save(output_html)
print(f"Nem haritanız başarıyla oluşturuldu!\nDosya Konumu: {output_html}")