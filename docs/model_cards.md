# Model Cards

## Overview

This document provides detailed information about the machine learning models
used in the Elite Set-Piece Analytics platform, following the Model Card format.

---

## First Receiver Predictor

### Model Details

| Property | Value |
|----------|-------|
| **Model Name** | ReceiverPredictor |
| **Version** | 0.1.0 |
| **Type** | Multi-class Classification |
| **Framework** | XGBoost + Scikit-learn |
| **Training Data** | Set-piece events with receiver labels |

### Intended Use

**Primary Use**: Predict the most likely first receiver of a set-piece delivery.

**Users**: Football analysts, coaches, data scientists, sports broadcasters.

**Out-of-Scope**: 
- Real-time in-game predictions (latency requirements)
- Player ability assessment

### Training Data

| Property | Value |
|----------|-------|
| Source | StatsBomb, Wyscout |
| Size | ~10,000 set-piece events |
| Time Period | 2018-2022 |
| Competitions | World Cup, Major Leagues |
| Label Distribution | Varies by match (5-15 receivers) |

### Model Architecture

```
Input Features (15-30 features)
        │
        ▼
┌───────────────────┐
│   StandardScaler  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    XGBoost        │
│   (200 trees,     │
│    depth 6)       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Probabilities   │
│   per receiver    │
└───────────────────┘
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 70-75% |
| Top-3 Accuracy | 85-90% |
| Top-5 Accuracy | 92-95% |
| AUC-ROC (macro) | 0.80-0.85 |
| Log Loss | 1.5-2.0 |

### Features Used

**Spatial Features**:
- Distance to goal
- Angle to goal
- Player density
- Zone classification
- Proximity to teammates/opponents

**Temporal Features**:
- Match time
- Sequence position
- Event duration

**Context Features**:
- Set-piece type
- Delivery type
- Score difference

### Limitations

1. **Class Imbalance**: Some receivers appear rarely
2. **Context Dependency**: Performance varies by team/league
3. **Data Availability**: Requires quality positional data
4. **Generalization**: May not transfer across different leagues

### Ethical Considerations

- Model should not be used for betting/gambling purposes
- Predictions are probabilistic, not deterministic
- Performance should be regularly validated

---

## Outcome Predictor

### Model Details

| Property | Value |
|----------|-------|
| **Model Name** | OutcomePredictor |
| **Version** | 0.1.0 |
| **Type** | Binary Classification |
| **Framework** | XGBoost |
| **Training Data** | Set-piece events with outcomes |

### Intended Use

**Primary Use**: Predict the probability of scoring from a set-piece.

**Users**: Analysts, coaches, broadcasters.

### Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.75-0.80 |
| Brier Score | 0.05-0.08 |
| Calibration | Good |

### Features Used

- Set-piece type
- Location on pitch
- Player positions
- Historical conversion rates

### Limitations

1. **Rare Events**: Goals from set-pieces are infrequent
2. **Calibration**: Requires careful calibration
3. **Feature Quality**: Depends on input data quality

---

## Pattern Analyzer

### Model Details

| Property | Value |
|----------|-------|
| **Model Name** | PatternAnalyzer |
| **Version** | 0.1.0 |
| **Type** | Unsupervised Clustering |
| **Framework** | K-Means + PCA |

### Intended Use

**Primary Use**: Identify and cluster similar set-piece patterns.

**Users**: Analysts, coaches for tactical preparation.

### Model Architecture

```
Input Features
      │
      ▼
┌─────────────────┐
│  StandardScaler │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      PCA        │
│ (95% variance)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    K-Means      │
│  (10 clusters)  │
└────────┬────────┘
         │
         ▼
   Cluster Labels
```

### Cluster Types

| Cluster | Description | Frequency |
|---------|-------------|-----------|
| 0 | Near-post corners | 15% |
| 1 | Far-post corners | 18% |
| 2 | Short corners | 8% |
| 3 | Direct free kicks | 12% |
| 4 | Indirect free kicks | 10% |
| ... | ... | ... |

### Limitations

1. **Fixed Clusters**: Number of clusters is predefined
2. **Interpretation**: Clusters need manual interpretation
3. **Stability**: Results may vary with different data

---

## Model Maintenance

### Retraining Schedule

- **Frequency**: After each major tournament
- **Trigger**: Performance drop > 5%
- **Data**: Include recent matches

### Version Control

- Models stored with version numbers
- Previous versions retained for comparison
- Changelog maintained for each update

### Monitoring

- Track prediction accuracy over time
- Monitor for distribution shift
- Log prediction confidence

---

## Contact

For questions about these models, please open an issue on the
[GitHub repository](https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics).
