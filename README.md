# Project Description
This repository provides an AI/ML-based price prediction system for various DeFi use cases (e.g., stablecoins, lending-borrowing, vaults, DEX). It features:
- Automated Data Pipelines – Collect, process, and store large volumes of asset data.
- Advanced Model Training – Leverages multiple models with random search and Bayesian optimization to systematically find and train top performers.
- Intelligent Caching – Minimizes unnecessary computations by caching and updating predictions only when needed.
- Comprehensive Monitoring – Ensures every pipeline and service is continuously tracked and maintained.
- MLflow Integration – Tracks experiments, enabling seamless retraining on fresh data and swift deployment of best-performing models.
- Containerized Infrastructure – Uses Docker to streamline development, deployment, and scalability while reducing bottlenecks and maximizing resource efficiency.

Use the instructions below to get started, manage infrastructure, and run individual services in Docker.

### Get started
1. Install Docker
2. Run the following command to create a Postgres database locally with Docker:
```
docker run -d \
  --name price-predict-postgres \
  -e POSTGRES_DB=postgres \
  -e POSTGRES_USER=dev \
  -e POSTGRES_PASSWORD=default \
  -p 5430:5432 \
postgres:16.2
```
Alternatively, you can use hosted database.

3. Run database migration to create needed tables:
```
docker build -t migration -f docker/Dockerfile.migration .
docker run --rm -e GENERATE_SCRIPTS=false migration
```
See [Migration Guide](./alembic/README) for more.

4. Create `.env` file from `.env.example` and fill required fields.
5. Build and run everything: 
```
docker compose -f docker/docker-compose.yml up -d --build
```
6. To stop everything:
```
docker compose -f docker/docker-compose.yml down
```

# Development
## Run individual services

### MLFlow server
```
docker build -t mlflow -f docker/Dockerfile.mlflow .
docker run -d --name mlflow -p 5555:5555 -v ./mlartifacts:/app/mlartifacts -v ./mlruns:/app/mlruns mlflow
```

### FastAPI server
```
docker build -t api -f docker/Dockerfile.api .
docker run -d --name api -p 8001:8001 --link mlflow api
```

### Data workers
```
docker build -t data-workers -f docker/Dockerfile.data .
docker run -d --name data-workers data-workers
```

### Model training
```
docker build -t model_training -f docker/Dockerfile.model_training .
docker run -d --name model_training --link mlflow model_training
```

### Cache worker
```
docker build -t cache -f docker/Dockerfile.cache .
docker run -d --name cache --link mlflow cache
```

### Monitoring worker
```
docker build -t monitoring -f docker/Dockerfile.monitoring .
docker run -d --name monitoring --link mlflow monitoring
```

### Portainer
```
docker volume create portainer_data
docker run -d -p 8000:8000 -p 9443:9443 --name portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:2.20.3
```

### Jupyter Notebook (need to run mlflow separately)
```
docker build -t jupyter -f docker/Dockerfile.jupyter .
docker run -d -p 8888:8888 -v ./:/app --name jupyter --link mlflow jupyter
```

# Other
### Configuring models
You can control which models are trained by editing the `models` section in `config.yaml`. Each entry defines a model name, the asset pair, and the timeframes for input data and prediction targets. When the system starts, it automatically generates corresponding API endpoints for each defined model and structures the entire training pipeline around these configurations. This way, you can easily add or remove models, as well as adjust timeframes, without needing to modify the core application code.

### Adding dependencies
When adding dependencies, don't forget to add them in `dependencies.txt` file.

### DB Migrations
Check [Migration Guide](./alembic/README) for database migration instructions.

### Coingecko API
To check the usage for Coingecko API:
https://docs.coingecko.com/reference/api-usage

### Docker monitoring
Follow this guide to setup Portainer for easy monitoring and interraction with docker containers: https://docs.portainer.io/start/install-ce/server/docker/linux

Use Portainer `:2.20.3` version or later to avoid bug when connecting to a console.