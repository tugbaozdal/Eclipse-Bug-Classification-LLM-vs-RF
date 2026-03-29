A Comparative Analysis of Traditional Machine Learning and Large Language Models in Software Bug Report Classification: A Case Study on the Eclipse Project
Bu çalışma, yazılım hata raporlarının otomatik sınıflandırılmasında Geleneksel Makine Öğrenmesi (Random Forest) ile Büyük Dil Modelleri (Gemini 2.5 Flash-Lite) arasındaki karşılaştırmalı bir analizi sunar. Eclipse Projesi'nden alınan dengelenmiş bir veri seti kullanılarak, teknik terminolojinin bu iki farklı yaklaşım tarafından nasıl işlendiği değerlendirilmiştir.

📝 Proje Özeti
Yazılım geliştirme süreçlerinde hata raporlarının doğru birimlere yönlendirilmesi kritik bir öneme sahiptir. Bu projede, Eclipse projesine ait hata raporları; Core, UI, Releng, Diagram ve SWT olmak üzere 5 farklı teknik birime sınıflandırılmıştır. Çalışma, spesifik teknik metinler üzerinde eğitilmiş bir modelin (RF) başarısını, genel amaçlı bir dil modelinin (LLM) "zero-shot" performansı ile kıyaslamaktadır.

🚀 Temel Bulgular
Random Forest: %64 doğruluk (accuracy) ve SWT gibi spesifik teknik birimlerde 0.82 gibi yüksek bir F1-skoru elde etmiştir.

Gemini LLM: "Zero-shot" sınıflandırma düzeninde %51 doğruluk göstermiştir; genelleme potansiyeli sunmakla birlikte, uzmanlaşmış teknik alanlar için ince ayar (fine-tuning) gereksinimi gözlemlenmiştir.

Optimal Hiper-parametreler: Yapılan deneyler sonucunda, Random Forest modelinde sınırsız ağaç derinliğinin (max_depth=None) karmaşık teknik örüntüleri yakalamada en iyi sonucu verdiği kanıtlanmıştır.

## 🚀 Temel Bulgular

- **Random Forest:** %64 doğruluk (accuracy) ve `SWT` gibi spesifik teknik birimlerde **0.82** gibi yüksek bir F1-skoru elde etmiştir.
- **Gemini LLM:** "Zero-shot" sınıflandırma düzeninde **%51** doğruluk göstermiştir.
- **Optimal Hiper-parametreler:** Yapılan deneyler sonucunda, Random Forest modelinde sınırsız ağaç derinliğinin (`max_depth=None`) karmaşık teknik örüntüleri yakalamada en iyi sonucu verdiği kanıtlanmıştır.

## 📂 Proje Yapısı

```text
KaliteProjesi/ (Ana Dizin)
├── README.md              # Proje tanıtım ve kullanım kılavuzu
├── requirements.txt       # Kütüphane bağımlılıkları listesi
├── .gitignore             # Git sistemine dahil edilmeyecek dosyalar
├── .env                   # API Anahtarları (Yerel dosya)
└── data_pipeline/         # Teknik çalışmaların bulunduğu klasör
    ├── data/              # Ham ve işlenmiş CSV veri setleri
    ├── figures/           # Analiz grafikleri (Confusion Matrix, Radar Chart vb.)
    ├── data_preprocessing.py # Veri hazırlama betiği
    ├── rf_classifier.py      # Random Forest süreci
    ├── gemini_classifier.py   # Gemini LLM mantığı
    └── visualize_results.py   # Grafik üreten araç


📂 Proje Yapısı (Project Structure)
Projenin temel işleyişi data_pipeline klasörü altında toplanmıştır. Aşağıda bu klasördeki dosyaların ve dizinlerin görevleri açıklanmaktadır:

📁 data_pipeline/
Bu dizin, veri hazırlama, model çalıştırma ve sonuçların görselleştirilmesi süreçlerini içerir.

📁 data/: Veri setlerinin ve model çıktılarını tutan klasör.

Eclipse.csv: Projede kullanılan ham veya temel veri seti.

train_data.csv / test_data.csv: Modelleri eğitmek ve test etmek için ayrılmış veri setleri.

rf_results.csv: Random Forest modelinin tahmin sonuçları.

gemini_final_results.csv: Gemini (LLM) modelinin türkçe promptla sınıflandırma sonuçları.

gemini_english_results.csv: Gemini modelinin İngilizce promptla veri seti üzerindeki sınıflandırma sonuçları.

📁 figures / final_figures: Modellerin performans metriklerini (Confusion Matrix, Accuracy grafikleri vb.) içeren görsel dosyalar.
figures klasöründe gemini'ye türkçe prompt vermiştik.
final_figures klasöründe ingilizce prompt verniştik.

🐍 data_preprocessing.py: Ham verileri temizlemek, normalize etmek ve modellerin işleyebileceği formata getirmek için kullanılan script.

🐍 rf_classifier.py: Random Forest algoritması ile hata sınıflandırması

🐍 gemini_classifier.py: Google Gemini API kullanarak hata sınıflandırması

🐍 gemini_classifier_en.py: Google Gemini API'de ingilizce prompt kullanarak hata sınıflandırması

🐍 visualize_results.py: Elde edilen CSV sonuçlarını kullanarak performans grafiklerini (karşılaştırmalı analizler) oluşturan script.

📄 Kök Dizindeki Diğer Dosyalar
.env: API anahtarları (Gemini API Key vb.) gibi hassas bilgileri tutan yapılandırma dosyası.

.gitignore: GitHub'a gönderilmemesi gereken dosyaların (örneğin .env veya büyük veri setleri) listesi.

requirements.txt: Projenin çalışması için gerekli olan Python kütüphanelerinin listesi (pandas, scikit-learn, google-generativeai vb.).

📊 Görselleştirmeler
Proje çıktısı olarak üretilen tüm performans karşılaştırmaları (Hata Matrisleri, Bar Grafikleri ve Model Yetenek Radar Grafiği) final_figures/ dizini altında yer almaktadır. Bu grafikler, modellerin hangi teknik birimlerde daha güçlü veya zayıf olduğunu görsel olarak sunar.

Bu proje akademik bir çalışma kapsamında karşılaştırmalı analiz amacıyla geliştirilmiştir.
```
