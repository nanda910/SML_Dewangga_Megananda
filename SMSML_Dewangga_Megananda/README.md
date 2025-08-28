# Proyek Akhir Machine Learning Dicoding - Skilled Level
# Dewangga Megananda

## Struktur Proyek

```
SMSML_Dewangga_Megananda/
├── Eksperimen_SML_Dewangga_Megananda/
│   └── preprocessing/
│       ├── Eksperimen_Dewangga_Megananda.ipynb
│       ├── automate_Dewangga_Megananda.py
│       ├── preprocessing.log
│       └── dataset_preprocessing/
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── *.log
│   ├── confusion_matrix*.png
│   ├── feature_importance*.csv
│   └── mlruns/
├── Workflow-CI/
│   ├── conda.yaml
│   ├── MLProject
│   └── .github/
└── Monitoring_dan_Logging/
    ├── inference.py
    ├── prometheus_exporter.py
    ├── prometheus.yml
    ├── *.log
    ├── grafana_dashboard.json
    └── screenshots/
```

## Cara Menjalankan

1. **Preprocessing**: Jalankan `automate_Dewangga_Megananda.py`
2. **Modelling**: Jalankan `modelling.py` kemudian `modelling_tuning.py`
3. **Inference API**: Jalankan `inference.py`
4. **Monitoring**: Jalankan `prometheus_exporter.py`

## Requirements

- Python 3.8+
- scikit-learn
- mlflow
- flask
- prometheus_client
- Lihat `requirements.txt` untuk daftar lengkap

## Screenshot Bukti

Screenshot hasil testing tersedia di folder `Monitoring_dan_Logging/screenshots/`

---
Dibuat oleh: Dewangga Megananda
Dicoding ID: [Masukkan Dicoding ID Anda]
