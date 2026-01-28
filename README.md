```bash
    uv sync
```

./scripts/deploy.sh pltv prod   

snow sql -q "CALL ML_LAYER_PLTV_DB.PROD.PLTV_RUN();"