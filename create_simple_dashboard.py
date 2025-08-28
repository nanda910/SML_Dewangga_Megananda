#!/usr/bin/env python3
"""
Simple Dashboard Creator for Grafana
Creates a basic dashboard with Prometheus metrics for Dicoding submission
"""

import json
import requests
import time

def create_simple_dashboard():
    """Create a simple dashboard that works with current Grafana version"""

    # Simple dashboard JSON compatible with Grafana 10+
    dashboard = {
        "dashboard": {
            "title": "Dewangga Megananda - ML Model Monitoring",
            "description": "Dashboard monitoring untuk model machine learning Dicoding Skilled Level",
            "tags": ["ml", "monitoring", "dicoding", "dewangga"],
            "timezone": "UTC",
            "panels": [
                {
                    "id": 1,
                    "title": "Model Accuracy",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "ml_model_accuracy{model_name=\"Dewangga_RF_Model\"}",
                            "legendFormat": "Accuracy"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percentunit",
                            "decimals": 2
                        }
                    },
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 0
                    }
                },
                {
                    "id": 2,
                    "title": "Model F1 Score",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "ml_model_f1_score{model_name=\"Dewangga_RF_Model\"}",
                            "legendFormat": "F1 Score"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percentunit",
                            "decimals": 2
                        }
                    },
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 0
                    }
                },
                {
                    "id": 3,
                    "title": "Request Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(ml_inference_requests_total{model_name=\"Dewangga_RF_Model\", status=\"success\"}[5m])",
                            "legendFormat": "Success Requests/sec"
                        }
                    ],
                    "gridPos": {
                        "h": 8,
                        "w": 24,
                        "x": 0,
                        "y": 8
                    }
                },
                {
                    "id": 4,
                    "title": "Response Time",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(ml_inference_request_duration_seconds_bucket{model_name=\"Dewangga_RF_Model\"}[5m]))",
                            "legendFormat": "95th Percentile"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "seconds"
                        }
                    },
                    "gridPos": {
                        "h": 8,
                        "w": 24,
                        "x": 0,
                        "y": 16
                    }
                },
                {
                    "id": 5,
                    "title": "Service Uptime",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "ml_model_uptime_seconds{model_name=\"Dewangga_RF_Model\"} / 3600",
                            "legendFormat": "Uptime Hours"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "hours",
                            "decimals": 1
                        }
                    },
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 24
                    }
                },
                {
                    "id": 6,
                    "title": "Total Requests",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "sum(ml_inference_requests_total{model_name=\"Dewangga_RF_Model\"})",
                            "legendFormat": "Total Requests"
                        }
                    ],
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 24
                    }
                }
            ],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "refresh": "30s",
            "schemaVersion": 38,
            "version": 1,
            "links": []
        }
    }

    # Save the dashboard
    with open('Monitoring_dan_Logging/simple_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)

    print("✅ Simple dashboard created: Monitoring_dan_Logging/simple_dashboard.json")
    print("\n📋 Manual Setup Instructions:")
    print("=" * 50)
    print("1. Open Grafana: http://localhost:3000")
    print("2. Login with admin/admin")
    print("3. Go to Data Sources:")
    print("   - Click the menu (☰) → Connections → Data sources")
    print("   - Click 'Add data source'")
    print("   - Search for 'Prometheus'")
    print("   - Set URL: http://localhost:9090")
    print("   - Click 'Save & Test'")
    print("\n4. Import Dashboard:")
    print("   - Click '+' → 'Import dashboard'")
    print("   - Upload: Monitoring_dan_Logging/simple_dashboard.json")
    print("   - Select your Prometheus data source")
    print("   - Click 'Import'")
    print("\n5. Take screenshots for Dicoding:")
    print("   - grafana_dashboard.jpg (the dashboard)")
    print("   - grafana_alert.jpg (Alerting → Alert rules)")

    return True

def test_prometheus_connection():
    """Test if Prometheus exporter is accessible"""
    try:
        response = requests.get("http://localhost:8001/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus exporter is running on port 8001")
            return True
        else:
            print(f"❌ Prometheus exporter returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Prometheus exporter: {e}")
        print("Make sure prometheus_exporter.py is running")
        return False

if __name__ == "__main__":
    print("Grafana Simple Dashboard Creator")
    print("For Dewangga Megananda ML Project")
    print("=" * 40)

    # Test services
    if not test_prometheus_connection():
        print("\n❌ Please start the Prometheus exporter first:")
        print("cd Monitoring_dan_Logging && python3 prometheus_exporter.py")
        exit(1)

    # Create dashboard
    create_simple_dashboard()

    print("\n🎉 Dashboard JSON created successfully!")
    print("Follow the manual setup instructions above.")