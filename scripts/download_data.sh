#!/bin/bash
# =============================================================================
# Elite Set-Piece Analytics - Data Download Script
# سكريبت تحميل البيانات
# =============================================================================

set -e

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
RAW_DIR="$DATA_DIR/raw"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}    Elite Set-Piece Analytics - Data Download     ${NC}"
echo -e "${GREEN}==================================================${NC}"

# Create directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p "$RAW_DIR/statsbomb"
mkdir -p "$RAW_DIR/wyscout"
mkdir -p "$RAW_DIR/metrica"
mkdir -p "$DATA_DIR/processed"

# =============================================================================
# StatsBomb Open Data
# =============================================================================
echo -e "\n${YELLOW}Downloading StatsBomb Open Data...${NC}"

if [ -d "$RAW_DIR/statsbomb/open-data" ]; then
    echo "StatsBomb data already exists. Updating..."
    cd "$RAW_DIR/statsbomb/open-data"
    git pull || echo "Could not update StatsBomb data"
    cd - > /dev/null
else
    echo "Cloning StatsBomb open-data repository..."
    git clone --depth 1 https://github.com/statsbomb/open-data.git "$RAW_DIR/statsbomb/open-data" || {
        echo -e "${RED}Failed to clone StatsBomb data${NC}"
        echo "You can manually download from: https://github.com/statsbomb/open-data"
    }
fi

# =============================================================================
# Metrica Sports Sample Data
# =============================================================================
echo -e "\n${YELLOW}Downloading Metrica Sports Sample Data...${NC}"

if [ -d "$RAW_DIR/metrica/sample-data" ]; then
    echo "Metrica data already exists. Updating..."
    cd "$RAW_DIR/metrica/sample-data"
    git pull || echo "Could not update Metrica data"
    cd - > /dev/null
else
    echo "Cloning Metrica sample-data repository..."
    git clone --depth 1 https://github.com/metrica-sports/sample-data.git "$RAW_DIR/metrica/sample-data" || {
        echo -e "${RED}Failed to clone Metrica data${NC}"
        echo "You can manually download from: https://github.com/metrica-sports/sample-data"
    }
fi

# =============================================================================
# Wyscout Event Data (if available)
# =============================================================================
echo -e "\n${YELLOW}Checking for Wyscout data...${NC}"

# Note: Wyscout data is typically proprietary
# The public dataset is available at:
# https://figshare.com/collections/Soccer_match_event_dataset/4415000

echo "Wyscout public data can be downloaded from:"
echo "  https://figshare.com/collections/Soccer_match_event_dataset/4415000"
echo ""
echo "To download manually:"
echo "  1. Visit the URL above"
echo "  2. Download the event data files"
echo "  3. Place them in: $RAW_DIR/wyscout/"

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}                 Download Summary                  ${NC}"
echo -e "${GREEN}==================================================${NC}"

echo -e "\nData directory: $DATA_DIR"
echo ""

# Check what was downloaded
if [ -d "$RAW_DIR/statsbomb/open-data" ]; then
    echo -e "${GREEN}✓${NC} StatsBomb Open Data"
else
    echo -e "${RED}✗${NC} StatsBomb Open Data"
fi

if [ -d "$RAW_DIR/metrica/sample-data" ]; then
    echo -e "${GREEN}✓${NC} Metrica Sample Data"
else
    echo -e "${RED}✗${NC} Metrica Sample Data"
fi

if [ -d "$RAW_DIR/wyscout" ] && [ "$(ls -A $RAW_DIR/wyscout 2>/dev/null)" ]; then
    echo -e "${GREEN}✓${NC} Wyscout Data"
else
    echo -e "${YELLOW}○${NC} Wyscout Data (manual download required)"
fi

echo -e "\n${GREEN}Done!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run 'pip install -r requirements.txt' to install dependencies"
echo "  2. Open notebooks/01_data_exploration.ipynb to explore the data"
echo ""
