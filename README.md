Project for asset product price predictions using different ML models

### Get started
1. Install Docker
2. Clone the project
3. Create `.env` file from `.env.example` and fill required values.
4. If there are currently running containers from docker compose, stop them with the following command:
```
docker compose -f docker/docker-compose.yml down
```
5. Build and run : 
```
docker compose -f docker/docker-compose.yml up -d --build
```


# Development

### Adding dependencies
When adding dependencies, don't forget to add them in `dependencies.txt` file.

## Database
Run the following command to create a Postgres database locally with Docker:
```
docker run -d \
  --name price-predict-postgres \
  -e POSTGRES_DB=postgres \
  -e POSTGRES_USER=dev \
  -e POSTGRES_PASSWORD=default \
  -p 5430:5432 \
postgres:16.2
```

When you just created a database, run database migration to create needed tables 
```
docker build -t migration -f docker/Dockerfile.migration .
docker run --rm -e GENERATE_SCRIPTS=false migration
```
See [Migration Guide](./alembic/README) for more.

## Running all
To run all services, you can use the same docker compose command as when running on production.

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

### Model training scripts
```
docker build -t models-btc -f docker/Dockerfile.models.btc .
docker run -d --name models-btc --link mlflow models-btc
```

```
docker build -t models-eth -f docker/Dockerfile.models.eth .
docker run -d --name models-eth --link mlflow models-eth
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

### Jupyter Notebook (for development) (need to run mlflow separately)
```
docker build -t jupyter -f docker/Dockerfile.jupyter .
docker run -d -p 8888:8888 -v ./:/app --name jupyter --link mlflow jupyter
```

# Other
### DB Migrations
Check `alembic/README.md` for database migration instructions.

### Coingecko API
To check the usage for Coingecko API:
https://docs.coingecko.com/reference/api-usage

### Docker monitoring
Follow this guide to setup Portainer for easy monitoring and interraction with docker containers: https://docs.portainer.io/start/install-ce/server/docker/linux

Use Portainer `:2.20.3` version or later to avoid bug when connecting to a console.