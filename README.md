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

📊 Görselleştirmeler
Proje çıktısı olarak üretilen tüm performans karşılaştırmaları (Hata Matrisleri, Bar Grafikleri ve Model Yetenek Radar Grafiği) figures/ dizini altında yer almaktadır. Bu grafikler, modellerin hangi teknik birimlerde daha güçlü veya zayıf olduğunu görsel olarak sunar.

Bu proje akademik bir çalışma kapsamında karşılaştırmalı analiz amacıyla geliştirilmiştir.
