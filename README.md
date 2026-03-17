```bash
    uv sync
```

./scripts/deploy.sh pltv prod
./scripts/deploy.sh pltv vbb
./scripts/deploy.sh adwords_gclid_upload prod


snow sql -q "CALL ML_LAYER_PLTV_DB.PROD.PLTV_RUN();"
