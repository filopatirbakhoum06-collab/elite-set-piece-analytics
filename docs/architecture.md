# Technical Architecture

## Overview

The Elite Set-Piece Analytics platform is a comprehensive sports analytics system
designed for analyzing football set-pieces, predicting first receivers, and
providing tactical intelligence.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Elite Set-Piece Analytics                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Data       │  │   Feature    │  │   ML         │  │   Dashboard  │    │
│  │   Layer      │  │   Layer      │  │   Layer      │  │   Layer      │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│         ▼                 ▼                 ▼                 ▼            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Core Processing Pipeline                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer (`src/data/`)

**Purpose**: Load, validate, and normalize data from multiple sources.

**Components**:
- `loaders.py`: Data loaders for Wyscout, StatsBomb, and Metrica
- `extractors.py`: Set-piece extraction and classification
- `preprocessors.py`: Data cleaning and validation

**Data Flow**:
```
Raw Data → Loaders → Normalizers → Extractors → Processed Data
```

### 2. Feature Layer (`src/features/`)

**Purpose**: Engineer features for machine learning models.

**Components**:
- `spatial.py`: Spatial features (distances, angles, zones)
- `temporal.py`: Time-based features (pressure, sequences)
- `physical.py`: Movement features (speed, acceleration)

**Feature Categories**:
| Category | Features | Description |
|----------|----------|-------------|
| Spatial | 10+ | Position-based calculations |
| Temporal | 5+ | Time and sequence features |
| Physical | 5+ | Movement and velocity |

### 3. ML Layer (`src/models/`)

**Purpose**: Train and deploy prediction models.

**Components**:
- `receiver_predictor.py`: First receiver prediction (XGBoost + optional NN)
- `outcome_predictor.py`: Set-piece outcome prediction (xG-like)
- `pattern_analyzer.py`: Tactical pattern clustering

**Model Architecture**:

```
                 ┌─────────────────┐
                 │   Input Features │
                 └────────┬────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │    XGBoost    │           │ Neural Network│
    │   Classifier  │           │   (optional)  │
    └───────┬───────┘           └───────┬───────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
                 ┌─────────────────┐
                 │    Ensemble     │
                 │   Predictions   │
                 └─────────────────┘
```

### 4. Visualization Layer (`src/visualization/`)

**Purpose**: Create static and interactive visualizations.

**Components**:
- `pitch.py`: Pitch drawing and player plotting
- `animations.py`: Tracking data animations
- `heatmaps.py`: Heat and density maps

### 5. Dashboard Layer (`dashboard/`)

**Purpose**: Provide interactive web interface.

**Components**:
- `app.py`: Main Streamlit application
- `pages/`: Individual dashboard pages
- `components/`: Reusable UI components

**Pages**:
1. Overview - Key metrics and summaries
2. Predictions - Interactive receiver prediction
3. Designer - Set-piece routine designer
4. Analysis - Deep dive analytics

## Data Sources

| Source | Data Type | Access |
|--------|-----------|--------|
| StatsBomb | Event data | Free API |
| Wyscout | Event data | Public dataset |
| Metrica | Tracking data | Sample data |

## API Reference

### ReceiverPredictor

```python
from src.models.receiver_predictor import ReceiverPredictor

predictor = ReceiverPredictor()
predictor.fit(X_train, y_train)
predictions = predictor.predict(X_test)
probabilities = predictor.predict_proba(X_test)
top_k = predictor.get_top_k_receivers(X_test, k=3)
```

### SpatialFeatures

```python
from src.features.spatial import SpatialFeatures

spatial = SpatialFeatures()
distance = spatial.distance_to_goal(x, y)
angle = spatial.angle_to_goal(x, y)
zone = spatial.classify_zone(x, y)
```

## Performance Considerations

1. **Data Loading**: Lazy loading for large datasets
2. **Feature Computation**: Vectorized operations with NumPy/Pandas
3. **Model Inference**: Batch predictions for efficiency
4. **Caching**: Dashboard caching for repeated queries

## Security

- No sensitive data stored in repository
- Environment variables for credentials
- Input validation on all user inputs

## Deployment

### Local Development

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

### Production

Recommended deployment options:
- Streamlit Cloud
- Docker container
- Cloud platforms (AWS, GCP, Azure)

## Future Enhancements

1. Real-time prediction API
2. Video integration
3. Advanced neural network models
4. Multi-language support
