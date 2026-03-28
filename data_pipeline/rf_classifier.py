import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# 1. Yapılandırma ve Dosya Yolları
TRAIN_DATA = 'data/train_data.csv'
TEST_DATA = 'data/test_data.csv'
OUTPUT_RESULTS = 'data/rf_results.csv'

def run_rf_pipeline():
    # 2. Veri Setlerini Yükle
    print("🚀 Veri setleri yükleniyor...")
    if not os.path.exists(TRAIN_DATA) or not os.path.exists(TEST_DATA):
        print("❌ Hata: CSV dosyaları bulunamadı!")
        return

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)

    # 3. Özellik Çıkarımı (TF-IDF Vektörizasyon)
    print("📊 Metin verileri sayısallaştırılıyor (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=2500)
    
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])

    y_train = train_df['co']
    y_test = test_df['co']

    # 4. Model Eğitimi
    # Deney sonucunda 'None' derinliğinin daha başarılı olduğu ispatlandığı için bu değer seçildi.
    print("🤖 Random Forest modeli eğitiliyor (max_depth=None)...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=None 
    )
    model.fit(X_train, y_train)

    # 5. Tahmin ve Değerlendirme
    print("🎯 Test süreci ve rapor oluşturma başlıyor...")
    y_pred = model.predict(X_test)

    print("\n" + "="*45)
    print("   RANDOM FOREST PERFORMANS RAPORU")
    print("="*45)
    print(classification_report(y_test, y_pred))

    # 6. Sonuçları Kaydet
    test_df['rf_prediction'] = y_pred
    test_df.to_csv(OUTPUT_RESULTS, index=False)
    print(f"\n✅ Güncellenmiş sonuçlar '{OUTPUT_RESULTS}' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_rf_pipeline()