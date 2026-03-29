import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# 1. Yapılandırma ve Dosya Yolları
RF_RESULTS = 'data/rf_results.csv'
GEMINI_RESULTS = 'data/gemini_english_results.csv'
FIGURES_DIR = 'final_figures/'

def generate_visualizations():
    # Klasör yoksa oluştur
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    # --- VERİLERİ YÜKLE ---
    print("📂 Veriler okunuyor...")
    df_rf = pd.read_csv(RF_RESULTS)
    df_gemini = pd.read_csv(GEMINI_RESULTS)
    
    labels = ['Core', 'Diagram', 'Releng', 'SWT', 'UI']
    
    # Gemini tahminlerini temizle
    df_gemini['prediction_cleaned'] = df_gemini['gemini_prediction'].apply(
        lambda x: x if x in labels else 'Other'
    )

    # Raporları hesapla (Skorları çekmek için)
    report_rf = classification_report(df_rf['co'], df_rf['rf_prediction'], output_dict=True)
    report_gemini = classification_report(df_gemini['actual_co'], df_gemini['prediction_cleaned'], output_dict=True)
    
    rf_f1 = [report_rf[label]['f1-score'] for label in labels]
    gemini_f1 = [report_gemini[label]['f1-score'] for label in labels]

    # --- 1. RANDOM FOREST CONFUSION MATRIX ---
    plt.figure(figsize=(10, 8))
    cm_rf = confusion_matrix(df_rf['co'], df_rf['rf_prediction'], labels=labels)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Random Forest - Confusion Matrix (Depth: None)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}rf_confusion_matrix.png')
    print("✅ RF Confusion Matrix kaydedildi.")

    # --- 2. GEMINI CONFUSION MATRIX ---
    plt.figure(figsize=(10, 8))
    cm_gemini = confusion_matrix(df_gemini['actual_co'], df_gemini['prediction_cleaned'], labels=labels)
    sns.heatmap(cm_gemini, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.title('Gemini 2.5 Flash-Lite - Confusion Matrix (English Prompt)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}gemini_confusion_matrix_en.png')
    print("✅ Gemini Confusion Matrix kaydedildi.")

    # --- 3. F1-SCORE BAR CHART ---
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, rf_f1, width, label='Random Forest', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, gemini_f1, width, label='Gemini 2.5 Flash-Lite', color='#e67e22', edgecolor='black')
    ax.set_ylabel('F1-Score')
    ax.set_title('Model Performance Comparison (F1-Score)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}f1_comparison_bar.png')
    print("✅ Karşılaştırma Bar Grafiği kaydedildi.")

    # --- 4. RADAR CHART (GÜNCEL!) ---
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    # Grafiği kapatmak için ilk değeri sona ekle
    rf_f1_radar = rf_f1 + [rf_f1[0]]
    gemini_f1_radar = gemini_f1 + [gemini_f1[0]]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Random Forest Alanı
    ax.fill(angles, rf_f1_radar, color='#3498db', alpha=0.25)
    ax.plot(angles, rf_f1_radar, color='#3498db', linewidth=2, label='Random Forest', marker='o')
    
    # Gemini Alanı
    ax.fill(angles, gemini_f1_radar, color='#e67e22', alpha=0.25)
    ax.plot(angles, gemini_f1_radar, color='#e67e22', linewidth=2, label='Gemini 2.5 Flash-Lite', marker='s')
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.0)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Model Capability Comparison (Radar Chart)', y=1.1)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}radar_chart_comparison.png')
    print("✅ Radar Chart 'figures/' klasörüne başarıyla kaydedildi.")

if __name__ == "__main__":
    generate_visualizations()
    print("\n🎉 Tebrikler! 4 harika grafik 'figures/' klasöründe seni bekliyor.")