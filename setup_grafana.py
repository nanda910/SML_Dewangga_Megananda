#!/usr/bin/env python3
"""
Grafana Setup Script - Dewangga Megananda
Sets up Grafana data source and imports dashboard programmatically
"""

import requests
import json
import time
import sys

def setup_grafana():
    """Setup Grafana data source and import dashboard"""

    base_url = "http://localhost:3000"
    auth = ("admin", "admin")

    print("🚀 Setting up Grafana for Dewangga Megananda ML Project...")
    print("=" * 60)

    # Step 1: Test Grafana connection
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Grafana is running!")
        else:
            print("❌ Grafana is not responding properly")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Grafana: {e}")
        print("Please make sure Grafana is running on http://localhost:3000")
        return False

    # Step 2: Create Prometheus data source
    print("\n📊 Creating Prometheus data source...")
    datasource_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://localhost:9090",
        "access": "browser",
        "isDefault": True,
        "jsonData": {
            "timeInterval": "15s",
            "queryTimeout": "60s"
        }
    }

    try:
        response = requests.post(
            f"{base_url}/api/datasources",
            json=datasource_payload,
            auth=auth
        )

        if response.status_code == 201:
            print("✅ Prometheus data source created successfully!")
            datasource_id = response.json()['datasource']['id']
        elif response.status_code == 409:
            print("✅ Prometheus data source already exists!")
            # Get existing datasource ID
            datasources = requests.get(f"{base_url}/api/datasources", auth=auth).json()
            for ds in datasources:
                if ds['name'] == 'Prometheus':
                    datasource_id = ds['id']
                    break
        else:
            print(f"❌ Failed to create data source: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating data source: {e}")
        return False

    # Step 3: Import dashboard
    print("\n📈 Importing ML monitoring dashboard...")

    try:
        # Load dashboard JSON
        with open('Monitoring_dan_Logging/grafana_dashboard.json', 'r') as f:
            dashboard_json = json.load(f)

        # Update datasource references
        for panel in dashboard_json['dashboard']['panels']:
            if 'targets' in panel:
                for target in panel['targets']:
                    target['datasource'] = {"type": "prometheus", "uid": "prometheus"}

        # Import dashboard
        import_payload = {
            "dashboard": dashboard_json['dashboard'],
            "overwrite": True,
            "inputs": [{
                "name": "DS_PROMETHEUS",
                "type": "datasource",
                "pluginId": "prometheus",
                "value": "Prometheus"
            }]
        }

        response = requests.post(
            f"{base_url}/api/dashboards/import",
            json=import_payload,
            auth=auth
        )

        if response.status_code == 200:
            dashboard_data = response.json()
            dashboard_url = dashboard_data['importedUrl']
            print("✅ Dashboard imported successfully!")
            print(f"🔗 Dashboard URL: {base_url}{dashboard_url}")
        else:
            print(f"❌ Failed to import dashboard: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error importing dashboard: {e}")
        return False

    # Specific error handling is done in the try block above

    print("\n🎉 Grafana setup completed successfully!")
    print("=" * 60)
    print("📋 Next steps:")
    print("1. Open your browser and go to: http://localhost:3000")
    print("2. Login with username: admin, password: admin")
    print("3. Navigate to the dashboard: Dewangga Megananda - ML Model Monitoring")
    print("4. Take screenshots for your Dicoding submission:")
    print("   - grafana_dashboard.jpg (the full dashboard)")
    print("   - grafana_alert.jpg (alert rules page)")

    return True

def check_services():
    """Check if all required services are running"""
    print("🔍 Checking service status...")

    services = [
        ("Grafana", "http://localhost:3000/api/health"),
        ("Prometheus Exporter", "http://localhost:8001/metrics"),
        ("ML Inference API", "http://localhost:5001/health")
    ]

    all_running = True

    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Running")
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                all_running = False
        except requests.exceptions.RequestException:
            print(f"❌ {name}: Not accessible")
            all_running = False

    return all_running

if __name__ == "__main__":
    print("Grafana Setup Script for Dewangga Megananda ML Project")
    print("This script will automatically configure Grafana with your dashboard")

    # Check services first
    if not check_services():
        print("\n⚠️  Some services are not running. Please start them first:")
        print("1. python3 Monitoring_dan_Logging/prometheus_exporter.py")
        print("2. python3 Monitoring_dan_Logging/inference.py")
        print("3. brew services start grafana")
        sys.exit(1)

    # Setup Grafana
    success = setup_grafana()

    if success:
        print("\n🎊 Setup completed! Your Grafana dashboard is ready.")
        print("Take your screenshots and submit to Dicoding! 🚀")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)