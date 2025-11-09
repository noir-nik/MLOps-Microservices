#!/bin/bash

echo "Starting ML Microservices Platform..."
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Titanic dataset exists
if [ ! -f "collector/data/titanic.csv" ]; then
    echo "Downloading Titanic dataset..."
    mkdir -p collector/data
    curl -o collector/data/titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
    echo "Dataset downloaded successfully!"
fi

# Create necessary directories
mkdir -p collector/logs
mkdir -p storage/logs
mkdir -p ml_service/logs
mkdir -p web_master/logs

# Build and start services
echo ""
echo "Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check health
echo ""
echo "Checking service health..."
curl -s http://localhost:8000/health | python -m json.tool

echo ""
echo "======================================"
echo "ML Microservices Platform is running!"
echo "======================================"
echo ""
echo "Access the services at:"
echo "  Frontend:    http://localhost:5000"
echo "  Web Master:  http://localhost:8000"
echo "  Collector:   http://localhost:8001"
echo "  Storage:     http://localhost:8002"
echo "  ML Service:  http://localhost:8003"
echo ""
echo "View logs with: docker-compose logs -f [service_name]"
echo "Stop services with: docker-compose down"
echo ""
