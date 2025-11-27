# Azure Container Apps Deployment Guide

This guide explains how to deploy the Deep RAG API to Azure Container Apps.

## Prerequisites

1. **Azure CLI** installed and configured
   ```bash
   az --version
   az login
   ```

2. **Docker** installed and running
   ```bash
   docker --version
   ```

3. **Environment Variables** set:
   ```bash
   export AZURE_OPENAI_API_KEY="your-key"
   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
   export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment"
   export TAVILY_API_KEY="your-key"  # Optional
   ```

## Quick Deployment

1. **Run the deployment script:**
   ```bash
   cd azure
   ./deploy.sh
   ```

   The script will:
   - Create a resource group
   - Create an Azure Container Registry (ACR)
   - Build and push the Docker image
   - Create a Container Apps environment
   - Deploy the container app

2. **Access your API:**
   - The script will output the FQDN (Fully Qualified Domain Name)
   - API will be available at: `https://<fqdn>`
   - Documentation at: `https://<fqdn>/docs`
   - Health check at: `https://<fqdn>/health`

## Manual Deployment Steps

### 1. Create Resource Group

```bash
az group create \
    --name deep-rag-rg \
    --location eastus
```

### 2. Create Azure Container Registry

```bash
az acr create \
    --resource-group deep-rag-rg \
    --name deepragacr \
    --sku Basic \
    --admin-enabled true
```

### 3. Build and Push Docker Image

```bash
# Login to ACR
az acr login --name deepragacr

# Build image
docker build -t deepragacr.azurecr.io/deep-rag-api:latest .

# Push image
docker push deepragacr.azurecr.io/deep-rag-api:latest
```

### 4. Create Container Apps Environment

```bash
az containerapp env create \
    --name deep-rag-env \
    --resource-group deep-rag-rg \
    --location eastus
```

### 5. Create Container App

```bash
az containerapp create \
    --name deep-rag-api \
    --resource-group deep-rag-rg \
    --environment deep-rag-env \
    --image deepragacr.azurecr.io/deep-rag-api:latest \
    --target-port 8000 \
    --ingress external \
    --cpu 1.0 \
    --memory 2Gi \
    --min-replicas 1 \
    --max-replicas 5 \
    --registry-server deepragacr.azurecr.io \
    --registry-username <acr-username> \
    --registry-password <acr-password>
```

### 6. Set Environment Variables and Secrets

```bash
az containerapp update \
    --name deep-rag-api \
    --resource-group deep-rag-rg \
    --set-env-vars \
        "API_HOST=0.0.0.0" \
        "API_PORT=8000" \
        "LOG_LEVEL=INFO" \
        "LLM_PROVIDER=azure_openai" \
        "EMBEDDING_PROVIDER=azure_openai" \
    --set-secrets \
        "azure-openai-api-key=<your-key>" \
        "azure-openai-endpoint=<your-endpoint>" \
        "azure-openai-deployment-name=<your-deployment>" \
    --secret-env-vars \
        "AZURE_OPENAI_API_KEY=azure-openai-api-key" \
        "AZURE_OPENAI_ENDPOINT=azure-openai-endpoint" \
        "AZURE_OPENAI_DEPLOYMENT_NAME=azure-openai-deployment-name"
```

## Cost Optimization

Azure Container Apps uses **pay-per-use** pricing:

- **Consumption Plan**: Pay only for what you use
- **CPU**: $0.000012/vCPU-second
- **Memory**: $0.0000015/GB-second
- **Requests**: Included in base cost

**Estimated Monthly Cost** (with 1 replica, 1 CPU, 2GB RAM, minimal traffic):
- Base: ~$10-20/month
- With moderate traffic: ~$30-50/month

**Tips to reduce costs:**
1. Set `min-replicas` to 0 if you can tolerate cold starts
2. Use smaller CPU/memory allocations if possible
3. Monitor and adjust `max-replicas` based on actual usage
4. Use Azure Monitor to track costs

## Monitoring

### View Logs

```bash
az containerapp logs show \
    --name deep-rag-api \
    --resource-group deep-rag-rg \
    --follow
```

### View Metrics

```bash
az containerapp show \
    --name deep-rag-api \
    --resource-group deep-rag-rg \
    --query properties.template.scale
```

### Azure Portal

1. Navigate to your Container App in Azure Portal
2. Go to "Monitoring" section
3. View metrics, logs, and insights

## Updating the Deployment

### Update Image

```bash
# Rebuild and push
docker build -t deepragacr.azurecr.io/deep-rag-api:latest .
docker push deepragacr.azurecr.io/deep-rag-api:latest

# Update container app
az containerapp update \
    --name deep-rag-api \
    --resource-group deep-rag-rg \
    --image deepragacr.azurecr.io/deep-rag-api:latest
```

## Troubleshooting

### Container Not Starting

1. Check logs:
   ```bash
   az containerapp logs show --name deep-rag-api --resource-group deep-rag-rg
   ```

2. Verify environment variables:
   ```bash
   az containerapp show --name deep-rag-api --resource-group deep-rag-rg --query properties.template.containers[0].env
   ```

### Health Check Failing

1. Check if the service is initializing:
   - Initialization can take 1-2 minutes
   - Check logs for embedding loading progress

2. Verify API keys are set correctly

### High Costs

1. Check replica count:
   ```bash
   az containerapp show --name deep-rag-api --resource-group deep-rag-rg --query properties.template.scale
   ```

2. Reduce `min-replicas` if possible
3. Adjust CPU/memory allocations

## Cleanup

To delete all resources:

```bash
az group delete --name deep-rag-rg --yes --no-wait
```

## Additional Resources

- [Azure Container Apps Documentation](https://docs.microsoft.com/azure/container-apps/)
- [Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- [Container Apps Best Practices](https://docs.microsoft.com/azure/container-apps/best-practices)

