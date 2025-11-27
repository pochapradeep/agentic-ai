#!/bin/bash
# Deployment script for Azure Container Apps

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="deep-rag-rg"
LOCATION="eastus"
CONTAINER_APP_NAME="deep-rag-api"
ENVIRONMENT_NAME="deep-rag-env"
REGISTRY_NAME="${RESOURCE_GROUP}acr"
IMAGE_NAME="${REGISTRY_NAME}.azurecr.io/${CONTAINER_APP_NAME}:latest"

echo -e "${GREEN}Azure Container Apps Deployment Script${NC}"
echo "=========================================="

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Error: Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if logged in
echo -e "${YELLOW}Checking Azure login...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Not logged in. Please log in...${NC}"
    az login
fi

# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo -e "${GREEN}Using subscription: ${SUBSCRIPTION_ID}${NC}"

# Create resource group
echo -e "${YELLOW}Creating resource group...${NC}"
az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --output none || echo "Resource group already exists"

# Create Azure Container Registry
echo -e "${YELLOW}Creating Azure Container Registry...${NC}"
az acr create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${REGISTRY_NAME}" \
    --sku Basic \
    --admin-enabled true \
    --output none || echo "ACR already exists"

# Get ACR login credentials
echo -e "${YELLOW}Getting ACR credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name "${REGISTRY_NAME}" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "${REGISTRY_NAME}" --query passwords[0].value -o tsv)

# Login to ACR
echo -e "${YELLOW}Logging in to ACR...${NC}"
docker login "${REGISTRY_NAME}.azurecr.io" -u "${ACR_USERNAME}" -p "${ACR_PASSWORD}"

# Build and push Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
cd "$(dirname "$0")/.."
docker build -t "${IMAGE_NAME}" .

echo -e "${YELLOW}Pushing Docker image to ACR...${NC}"
docker push "${IMAGE_NAME}"

# Create Container Apps environment
echo -e "${YELLOW}Creating Container Apps environment...${NC}"
az containerapp env create \
    --name "${ENVIRONMENT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --output none || echo "Environment already exists"

# Create or update Container App
echo -e "${YELLOW}Creating/updating Container App...${NC}"

# Check if container app exists
if az containerapp show --name "${CONTAINER_APP_NAME}" --resource-group "${RESOURCE_GROUP}" &> /dev/null; then
    echo -e "${YELLOW}Updating existing container app...${NC}"
    az containerapp update \
        --name "${CONTAINER_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --image "${IMAGE_NAME}" \
        --output none
else
    echo -e "${YELLOW}Creating new container app...${NC}"
    az containerapp create \
        --name "${CONTAINER_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --environment "${ENVIRONMENT_NAME}" \
        --image "${IMAGE_NAME}" \
        --target-port 8000 \
        --ingress external \
        --cpu 1.0 \
        --memory 2Gi \
        --min-replicas 1 \
        --max-replicas 5 \
        --registry-server "${REGISTRY_NAME}.azurecr.io" \
        --registry-username "${ACR_USERNAME}" \
        --registry-password "${ACR_PASSWORD}" \
        --env-vars \
            "API_HOST=0.0.0.0" \
            "API_PORT=8000" \
            "LOG_LEVEL=INFO" \
            "LOG_FORMAT=json" \
            "LLM_PROVIDER=azure_openai" \
            "EMBEDDING_PROVIDER=azure_openai" \
        --secrets \
            "azure-openai-api-key=${AZURE_OPENAI_API_KEY}" \
            "azure-openai-endpoint=${AZURE_OPENAI_ENDPOINT}" \
            "azure-openai-deployment-name=${AZURE_OPENAI_DEPLOYMENT_NAME}" \
            "tavily-api-key=${TAVILY_API_KEY}" \
        --secret-env-vars \
            "AZURE_OPENAI_API_KEY=azure-openai-api-key" \
            "AZURE_OPENAI_ENDPOINT=azure-openai-endpoint" \
            "AZURE_OPENAI_DEPLOYMENT_NAME=azure-openai-deployment-name" \
            "TAVILY_API_KEY=tavily-api-key" \
        --output none
fi

# Get the FQDN
FQDN=$(az containerapp show \
    --name "${CONTAINER_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo -e "${GREEN}=========================================="
echo -e "Deployment completed successfully!${NC}"
echo -e "${GREEN}=========================================="
echo ""
echo -e "Container App URL: ${GREEN}https://${FQDN}${NC}"
echo -e "API Documentation: ${GREEN}https://${FQDN}/docs${NC}"
echo -e "Health Check: ${GREEN}https://${FQDN}/health${NC}"
echo ""
echo -e "${YELLOW}Note: Make sure to set the following environment variables:${NC}"
echo "  - AZURE_OPENAI_API_KEY"
echo "  - AZURE_OPENAI_ENDPOINT"
echo "  - AZURE_OPENAI_DEPLOYMENT_NAME"
echo "  - TAVILY_API_KEY (optional)"
echo ""

