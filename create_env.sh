#!/bin/bash
set -e

user_email="$1"
sub="$2"
location="southindia"
rgname=$(echo "$user_email" | cut -d "@" -f1)
git_repo="https://github.com/Jeganm91/ContextEngineeringVersion03.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

az group create -l "$location" -n "$rgname" --subscription "$sub" > /dev/null

id=$(az ad user show --id "$1" --query "id" --output tsv)
subid=$(az account show --name "$sub" --query id -o tsv)
scope="/subscriptions/$subid/resourceGroups/$rgname"

rand=$(date +%s | tail -c 6)
stname="ragst${rand}"
srchname="ragsrch${rand}"
oiname="ragoi${rand}"
idxname="rag-index-${rand}"
dsname="rag-ds-${rand}"
skname="rag-skillset-${rand}"
idxrname="rag-indexer-${rand}"

# --- role assignments + policy, background, non-blocking ---
az role assignment create --assignee "$id" --role "DenyPolicyDelete" --scope "$scope" > /dev/null 2>&1 &
az role assignment create --assignee "$id" --role "Storage Blob Data Contributor" --scope "$scope" > /dev/null 2>&1 &
az role assignment create --assignee "$id" --role "Search Index Data Contributor" --scope "$scope" > /dev/null 2>&1 &
az role assignment create --assignee "$id" --role "Search Service Contributor" --scope "$scope" > /dev/null 2>&1 &
az role assignment create --assignee "$id" --role "Cognitive Services OpenAI Contributor" --scope "$scope" > /dev/null 2>&1 &

# --- Search + OpenAI creation, background, overlaps VM+Storage ---
az search service create -n "$srchname" -g "$rgname" -l "$location" --sku Basic --partition-count 1 --replica-count 1 --subscription "$sub" > /dev/null 2>&1 &
SEARCH_PID=$!
az cognitiveservices account create -n "$oiname" -g "$rgname" -l "$location" --kind OpenAI --sku S0 --subscription "$sub" > /dev/null 2>&1 &
OPENAI_PID=$!

# --- Storage: on critical path ---
az storage account create -n "$stname" -g "$rgname" -l "$location" --sku Standard_LRS --kind StorageV2 --subscription "$sub" > /dev/null
stkey=$(az storage account keys list -n "$stname" -g "$rgname" --subscription "$sub" --query "[0].value" -o tsv)
az storage container create -n "kb-docs" --account-name "$stname" --account-key "$stkey" --subscription "$sub" --only-show-errors > /dev/null

# --- Upload knowledge docs WITH metadata (doc_id/status/effective_date) so the
#     indexer can map them straight to index fields without a custom parser ---
for f in "${SCRIPT_DIR}"/../knowledge_base_docs/*.md; do
    fname=$(basename "$f")
    doc_id=$(grep -m1 "^doc_id:" "$f" | sed 's/doc_id:[[:space:]]*//')
    status=$(grep -m1 "^status:" "$f" | sed 's/status:[[:space:]]*//')
    eff_date=$(grep -m1 "^effective_date:" "$f" | sed 's/effective_date:[[:space:]]*//')
    az storage blob upload --account-name "$stname" --account-key "$stkey" \
        --container-name kb-docs --file "$f" --name "$fname" \
        --metadata doc_id="$doc_id" status="$status" effective_date="$eff_date" \
        --only-show-errors --overwrite > /dev/null
done

wait $OPENAI_PID
# --- Model deployments (needs OpenAI resource to exist first) ---
az cognitiveservices account deployment create --name "$oiname" -g "$rgname" --subscription "$sub" \
    --deployment-name "text-embedding-3-small" --model-name "text-embedding-3-small" \
    --model-version "1" --model-format OpenAI --sku-capacity 10 --sku-name "Standard" > /dev/null
az cognitiveservices account deployment create --name "$oiname" -g "$rgname" --subscription "$sub" \
    --deployment-name "gpt-5-mini" --model-name "gpt-5-mini" \
    --model-version "1" --model-format OpenAI --sku-capacity 10 --sku-name "Standard" > /dev/null

oi_endpoint=$(az cognitiveservices account show -n "$oiname" -g "$rgname" --subscription "$sub" --query "properties.endpoint" -o tsv)
oi_key=$(az cognitiveservices account keys list -n "$oiname" -g "$rgname" --subscription "$sub" --query "key1" -o tsv)

wait $SEARCH_PID
srch_key=$(az search admin-key show --resource-group "$rgname" --service-name "$srchname" --subscription "$sub" --query "primaryKey" -o tsv)
srch_endpoint="https://${srchname}.search.windows.net"

# ---------------------------------------------------------------------------
# Build the retrieval pipeline entirely via REST -- index, data source,
# skillset (chunk + embed), indexer. Participants never touch this.
# ---------------------------------------------------------------------------
curl -s -X PUT "${srch_endpoint}/indexes/${idxname}?api-version=2024-07-01" \
  -H "api-key: ${srch_key}" -H "Content-Type: application/json" -d @- > /dev/null <<EOF
{
  "name": "${idxname}",
  "fields": [
    {"name": "chunk_id", "type": "Edm.String", "key": true, "analyzer": "keyword"},
    {"name": "content", "type": "Edm.String", "searchable": true},
    {"name": "title", "type": "Edm.String", "searchable": true, "filterable": true},
    {"name": "doc_id", "type": "Edm.String", "filterable": true, "facetable": true},
    {"name": "status", "type": "Edm.String", "filterable": true, "facetable": true},
    {"name": "effective_date", "type": "Edm.String", "filterable": true, "sortable": true},
    {"name": "text_vector", "type": "Collection(Edm.Single)", "searchable": true,
     "dimensions": 1536, "vectorSearchProfile": "vec-profile"}
  ],
  "vectorSearch": {
    "algorithms": [{"name": "hnsw-cfg", "kind": "hnsw"}],
    "profiles": [{"name": "vec-profile", "algorithm": "hnsw-cfg", "vectorizer": "aoai-vectorizer"}],
    "vectorizers": [{
      "name": "aoai-vectorizer", "kind": "azureOpenAI",
      "azureOpenAIParameters": {
        "resourceUri": "${oi_endpoint}", "deploymentId": "text-embedding-3-small",
        "apiKey": "${oi_key}", "modelName": "text-embedding-3-small"
      }
    }]
  }
}
EOF

curl -s -X PUT "${srch_endpoint}/datasources/${dsname}?api-version=2024-07-01" \
  -H "api-key: ${srch_key}" -H "Content-Type: application/json" -d @- > /dev/null <<EOF
{
  "name": "${dsname}",
  "type": "azureblob",
  "credentials": {"connectionString": "DefaultEndpointsProtocol=https;AccountName=${stname};AccountKey=${stkey};EndpointSuffix=core.windows.net"},
  "container": {"name": "kb-docs"}
}
EOF

curl -s -X PUT "${srch_endpoint}/skillsets/${skname}?api-version=2024-07-01" \
  -H "api-key: ${srch_key}" -H "Content-Type: application/json" -d @- > /dev/null <<EOF
{
  "name": "${skname}",
  "skills": [{
    "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
    "context": "/document",
    "resourceUri": "${oi_endpoint}", "deploymentId": "text-embedding-3-small",
    "apiKey": "${oi_key}", "modelName": "text-embedding-3-small",
    "inputs": [{"name": "text", "source": "/document/content"}],
    "outputs": [{"name": "embedding", "targetName": "text_vector"}]
  }]
}
EOF

curl -s -X PUT "${srch_endpoint}/indexers/${idxrname}?api-version=2024-07-01" \
  -H "api-key: ${srch_key}" -H "Content-Type: application/json" -d @- > /dev/null <<EOF
{
  "name": "${idxrname}",
  "dataSourceName": "${dsname}",
  "targetIndexName": "${idxname}",
  "skillsetName": "${skname}",
  "fieldMappings": [
    {"sourceFieldName": "metadata_storage_path", "targetFieldName": "chunk_id", "mappingFunction": {"name": "base64Encode"}},
    {"sourceFieldName": "metadata_storage_name", "targetFieldName": "title"},
    {"sourceFieldName": "metadata_doc_id", "targetFieldName": "doc_id"},
    {"sourceFieldName": "metadata_status", "targetFieldName": "status"},
    {"sourceFieldName": "metadata_effective_date", "targetFieldName": "effective_date"},
    {"sourceFieldName": "content", "targetFieldName": "content"}
  ],
  "outputFieldMappings": [
    {"sourceFieldName": "/document/text_vector", "targetFieldName": "text_vector"}
  ]
}
EOF

curl -s -X POST "${srch_endpoint}/indexers/${idxrname}/run?api-version=2024-07-01" -H "api-key: ${srch_key}" > /dev/null

# ---------------------------------------------------------------------------
# Cloud-init: env file is fully populated. Participants only ever edit
# application code, never Azure resources.
# ---------------------------------------------------------------------------
cloud_init_content=$(cat <<EOF
#cloud-config
package_upgrade: false
packages: [python3-pip, python3-venv, git, curl, jq]
runcmd:
  - git clone ${git_repo} /opt/rag-lab
  - python3 -m venv /opt/rag-lab/.venv
  - /opt/rag-lab/.venv/bin/pip install --upgrade pip
  - /opt/rag-lab/.venv/bin/pip install flask "openai>=1.55.3" azure-search-documents azure-core requests
  - mkdir -p /etc/environment.d
  - |
    cat <<EOC > /etc/environment.d/rag-lab.conf
    AZURE_SEARCH_SERVICE_ENDPOINT=${srch_endpoint}
    AZURE_SEARCH_INDEX_NAME=${idxname}
    AZURE_SEARCH_API_KEY=${srch_key}
    AZURE_OPENAI_ENDPOINT=${oi_endpoint}
    AZURE_OPENAI_API_KEY=${oi_key}
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
    VECTOR_FIELD_NAME=text_vector
    MCP_SERVER_URL=http://localhost:9001
    EOC
  - |
    cat <<EOC > /etc/systemd/system/rag-app.service
    [Unit]
    Description=RAG Lab Flask App
    After=network.target
    [Service]
    WorkingDirectory=/opt/rag-lab/app
    EnvironmentFile=/etc/environment.d/rag-lab.conf
    ExecStart=/opt/rag-lab/.venv/bin/python /opt/rag-lab/app/app.py
    Restart=always
    User=root
    [Install]
    WantedBy=multi-user.target
    EOC
  - |
    cat <<EOC > /etc/systemd/system/rag-mcp.service
    [Unit]
    Description=RAG Lab MCP Server
    After=network.target
    [Service]
    WorkingDirectory=/opt/rag-lab/mcp_server
    ExecStart=/opt/rag-lab/.venv/bin/python /opt/rag-lab/mcp_server/server.py
    Restart=always
    User=root
    [Install]
    WantedBy=multi-user.target
    EOC
  - systemctl daemon-reload
  - systemctl enable --now rag-mcp.service
  - systemctl enable --now rag-app.service
EOF
)
custom_data_b64=$(echo "$cloud_init_content" | base64 | tr -d '\n')

deploy_output=$(az deployment group create --name "infra-deploy-${rand}" --resource-group "$rgname" --subscription "$sub" \
  --template-file "${SCRIPT_DIR}/deploy_infra.json" --parameters customDataBase64="$custom_data_b64" \
  --query "properties.outputs" -o json)
vm_principal_id=$(echo "$deploy_output" | jq -r '.vmPrincipalId.value')

az role assignment create --assignee "$vm_principal_id" --role "Storage Blob Data Contributor" --scope "$scope" > /dev/null 2>&1 &
az role assignment create --assignee "$vm_principal_id" --role "Search Index Data Contributor" --scope "$scope" > /dev/null 2>&1 &
az role assignment create --assignee "$vm_principal_id" --role "Cognitive Services OpenAI Contributor" --scope "$scope" > /dev/null 2>&1 &

wait
echo "Success"
