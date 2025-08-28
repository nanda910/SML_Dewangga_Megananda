#!/usr/bin/env python3
"""
Prometheus Exporter - Dewangga Megananda
Proyek Akhir Machine Learning Dicoding - Skilled Level

Script untuk expose metrics ke Prometheus dengan minimal 5 metrik.
Jalan di localhost:8001/metrics
"""

import os
import sys
import time
import logging
import threading
from flask import Flask
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest,
    CONTENT_TYPE_LATEST, start_http_server
)
import requests
import psutil
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prometheus_exporter.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics - minimal 5 metrik
REQUEST_COUNT = Counter(
    'ml_inference_requests_total',
    'Total number of ML inference requests',
    ['model_name', 'status']
)

REQUEST_LATENCY = Histogram(
    'ml_inference_request_duration_seconds',
    'ML inference request duration in seconds',
    ['model_name']
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_name']
)

MODEL_F1_SCORE = Gauge(
    'ml_model_f1_score',
    'Current model F1 score',
    ['model_name']
)

MODEL_UPTIME = Gauge(
    'ml_model_uptime_seconds',
    'Model service uptime in seconds',
    ['model_name']
)

# Additional metrics
CPU_USAGE = Gauge('ml_service_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('ml_service_memory_usage_mb', 'Memory usage in MB')
DISK_USAGE = Gauge('ml_service_disk_usage_percent', 'Disk usage percentage')

# Prediction metrics
PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_name', 'predicted_class']
)

class MetricsCollector:
    """Class untuk collect dan update metrics"""

    def __init__(self, model_name="Dewangga_RF_Model"):
        self.model_name = model_name
        self.start_time = time.time()
        self.prediction_counts = {'setosa': 0, 'versicolor': 0, 'virginica': 0}

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024
            MEMORY_USAGE.set(memory_mb)

            # Disk usage
            disk = psutil.disk_usage('/')
            DISK_USAGE.set(disk.percent)

        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")

    def update_model_metrics(self):
        """Update model-specific metrics"""
        try:
            # Update uptime
            uptime = time.time() - self.start_time
            MODEL_UPTIME.labels(model_name=self.model_name).set(uptime)

            # Simulate model metrics (dalam praktik nyata, ambil dari model evaluation)
            # Ini bisa diintegrasikan dengan actual model performance
            accuracy = 0.95 + random.uniform(-0.02, 0.02)  # 93-97% accuracy
            f1_score = 0.94 + random.uniform(-0.02, 0.02)  # 92-96% F1

            MODEL_ACCURACY.labels(model_name=self.model_name).set(accuracy)
            MODEL_F1_SCORE.labels(model_name=self.model_name).set(f1_score)

        except Exception as e:
            logger.warning(f"Error updating model metrics: {e}")

    def record_prediction(self, predicted_class):
        """Record prediction untuk tracking"""
        if predicted_class in self.prediction_counts:
            self.prediction_counts[predicted_class] += 1
            PREDICTION_COUNT.labels(
                model_name=self.model_name,
                predicted_class=predicted_class
            ).inc()

    def simulate_inference_requests(self):
        """Simulate inference requests untuk demo metrics"""
        while True:
            try:
                # Simulate random requests
                if random.random() < 0.3:  # 30% chance setiap interval
                    # Simulate request latency
                    latency = random.uniform(0.1, 2.0)
                    REQUEST_LATENCY.labels(model_name=self.model_name).observe(latency)

                    # Simulate success/failure
                    if random.random() < 0.95:  # 95% success rate
                        REQUEST_COUNT.labels(model_name=self.model_name, status='success').inc()

                        # Simulate prediction
                        classes = ['setosa', 'versicolor', 'virginica']
                        predicted_class = random.choice(classes)
                        self.record_prediction(predicted_class)
                    else:
                        REQUEST_COUNT.labels(model_name=self.model_name, status='error').inc()

            except Exception as e:
                logger.warning(f"Error simulating requests: {e}")

            time.sleep(5)  # Update setiap 5 detik

class MLServiceExporter:
    """Main class untuk Prometheus exporter"""

    def __init__(self, port=8001, model_name="Dewangga_RF_Model"):
        self.port = port
        self.model_name = model_name
        self.collector = MetricsCollector(model_name)
        self.app = Flask(__name__)

        # Setup Flask routes
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes untuk metrics endpoint"""

        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return {
                'status': 'healthy',
                'service': 'ML Prometheus Exporter',
                'model': self.model_name,
                'timestamp': time.time()
            }, 200

        @self.app.route('/', methods=['GET'])
        def root():
            """Root endpoint dengan informasi service"""
            info = {
                'service': 'Dewangga Megananda ML Prometheus Exporter',
                'version': '1.0.0',
                'description': 'Prometheus exporter untuk ML model metrics',
                'endpoints': {
                    'GET /': 'Service information',
                    'GET /health': 'Health check',
                    'GET /metrics': 'Prometheus metrics'
                },
                'metrics': {
                    'ml_inference_requests_total': 'Total inference requests',
                    'ml_inference_request_duration_seconds': 'Request latency histogram',
                    'ml_model_accuracy': 'Model accuracy gauge',
                    'ml_model_f1_score': 'Model F1 score gauge',
                    'ml_model_uptime_seconds': 'Service uptime',
                    'ml_service_cpu_usage_percent': 'CPU usage',
                    'ml_service_memory_usage_mb': 'Memory usage',
                    'ml_service_disk_usage_percent': 'Disk usage',
                    'ml_predictions_total': 'Prediction counts by class'
                },
                'model_name': self.model_name
            }
            return info, 200

    def start_metrics_collection(self):
        """Start background threads untuk collect metrics"""

        def collect_system_metrics():
            """Thread untuk collect system metrics"""
            while True:
                self.collector.update_system_metrics()
                time.sleep(10)  # Update setiap 10 detik

        def collect_model_metrics():
            """Thread untuk collect model metrics"""
            while True:
                self.collector.update_model_metrics()
                time.sleep(30)  # Update setiap 30 detik

        # Start background threads
        threading.Thread(target=collect_system_metrics, daemon=True).start()
        threading.Thread(target=collect_model_metrics, daemon=True).start()
        threading.Thread(target=self.collector.simulate_inference_requests, daemon=True).start()

        logger.info("Metrics collection threads started")

    def run(self):
        """Run the Prometheus exporter"""
        try:
            logger.info(f"Starting Prometheus exporter on port {self.port}")
            logger.info(f"Metrics available at http://localhost:{self.port}/metrics")

            # Start metrics collection
            self.start_metrics_collection()

            # Start Flask app
            self.app.run(host='0.0.0.0', port=self.port, debug=False)

        except Exception as e:
            logger.error(f"Failed to start exporter: {e}")
            sys.exit(1)

def main():
    """Main function untuk menjalankan Prometheus exporter"""
    try:
        exporter = MLServiceExporter()
        exporter.run()
    except KeyboardInterrupt:
        logger.info("Exporter stopped by user")
    except Exception as e:
        logger.error(f"Exporter failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()