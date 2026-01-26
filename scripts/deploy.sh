#!/bin/bash
# =============================================================================
# ML Layer Deployment Script
# =============================================================================
# 
# Usage:
#   ./scripts/deploy.sh <project> <target>
#   ./scripts/deploy.sh pltv prod         # Deploy PLTV to PROD
#   ./scripts/deploy.sh pltv staging      # Deploy PLTV to STAGING
#   ./scripts/deploy.sh pltv dev          # Deploy PLTV to DEV
#
# The TARGET environment variable controls the schema:
#   - DEV: Uses DEV_{DEVELOPER} schema
#   - STAGING: Uses STAGING schema
#   - PROD: Uses PROD schema
#
# Prerequisites:
#   - Snowflake CLI installed (snow --version)
#   - Valid .snowflake/config.toml with connection settings
#   - Private key at .keys/rsa_key.p8
#   - .env file with TARGET and DEVELOPER (for DEV)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Set SNOWFLAKE_HOME to use project's .snowflake config
export SNOWFLAKE_HOME="${PROJECT_ROOT}/.snowflake"

# Fix permissions on config.toml (Snowflake CLI requires 0600)
chmod 0600 "${SNOWFLAKE_HOME}/config.toml" 2>/dev/null || true

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Parse arguments
PROJECT=$(echo "$1" | tr '[:upper:]' '[:lower:]')
TARGET_ARG=$(echo "$2" | tr '[:lower:]' '[:upper:]')

# Validate PROJECT is provided
if [ -z "$PROJECT" ]; then
    echo -e "${RED}Error: PROJECT is required${NC}"
    echo "Usage: ./scripts/deploy.sh <project> <target>"
    echo "Example: ./scripts/deploy.sh pltv prod"
    exit 1
fi

# Override TARGET if argument provided, otherwise use .env
if [ -n "$TARGET_ARG" ]; then
    export TARGET="$TARGET_ARG"
fi

# Validate TARGET is set
if [ -z "$TARGET" ]; then
    echo -e "${RED}Error: TARGET is not set. Provide as argument or set in .env${NC}"
    echo "Usage: ./scripts/deploy.sh <project> <target>"
    exit 1
fi

# Validate project SQL file exists
SQL_FILE="setup/${PROJECT}/tasks.sql"
if [ ! -f "$SQL_FILE" ]; then
    echo -e "${RED}Error: SQL file not found: ${SQL_FILE}${NC}"
    echo "Available projects:"
    ls -d setup/*/ 2>/dev/null | xargs -I {} basename {} || echo "  (none found)"
    exit 1
fi

# Set database based on project (matches Project enum in app/projects/__init__.py)
PROJECT_UPPER=$(echo "$PROJECT" | tr '[:lower:]' '[:upper:]')
if [ "$PROJECT_UPPER" = "CORE" ]; then
    DATABASE="ML_LAYER_DB"
else
    DATABASE="ML_LAYER_${PROJECT_UPPER}_DB"
fi
STAGE="${DATABASE}.${TARGET}.ML_LAYER_STAGE"

echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}ML Layer Deployment${NC}"
echo -e "${YELLOW}=========================================${NC}"
echo ""
echo -e "Project: ${GREEN}${PROJECT_UPPER}${NC}"
echo -e "Target: ${GREEN}${TARGET}${NC}"
echo -e "Database: ${GREEN}${DATABASE}${NC}"
echo -e "Stage: ${GREEN}${STAGE}${NC}"
echo -e "SQL File: ${GREEN}${SQL_FILE}${NC}"
echo ""

# Step 1: Test connection
echo -e "${YELLOW}[1/5] Testing Snowflake connection...${NC}"
if snow connection test; then
    echo -e "${GREEN}Connection successful!${NC}"
else
    echo -e "${RED}Connection failed. Please check your config.toml and private key.${NC}"
    exit 1
fi
echo ""

# Step 2: Build artifacts
echo -e "${YELLOW}[2/5] Building Snowpark artifacts...${NC}"
snow snowpark build --ignore-anaconda
echo -e "${GREEN}Build successful!${NC}"
echo ""

# Step 3: Create stage if not exists
echo -e "${YELLOW}[3/5] Creating stage...${NC}"
snow sql -q "CREATE SCHEMA IF NOT EXISTS ${DATABASE}.${TARGET}"
snow sql -q "CREATE STAGE IF NOT EXISTS ${STAGE}"
echo -e "${GREEN}Stage ready!${NC}"
echo ""

# Step 4: Upload artifacts to stage
echo -e "${YELLOW}[4/5] Uploading artifacts to stage...${NC}"
snow stage copy app.zip "@${STAGE}/app/" --overwrite
echo -e "${GREEN}Artifacts uploaded!${NC}"
echo ""

# Step 5: Create procedure and tasks
echo -e "${YELLOW}[5/5] Creating procedure and tasks...${NC}"
snow sql -f "$SQL_FILE"
echo -e "${GREEN}Procedure and tasks configured!${NC}"
echo ""

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "To verify:"
echo "  export SNOWFLAKE_HOME=${SNOWFLAKE_HOME}"
echo "  snow sql -q \"SHOW PROCEDURES IN ${DATABASE}.${TARGET}\""
echo "  snow sql -q \"SHOW TASKS IN ${DATABASE}.${TARGET}\""
echo ""
echo "To execute task manually:"
echo "  snow sql -q \"EXECUTE TASK ${DATABASE}.${TARGET}.${PROJECT_UPPER}_WEEKLY_TASK\""
