# 🚀 Quick Start Guide

Get results in 5 minutes with the Elite Set-Piece Analytics Platform!

---

## Step 1: Clone Repository

```bash
git clone https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics.git
cd elite-set-piece-analytics
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Verify Installation

```bash
python -c "from src.data.loaders import load_wyscout_data; print('✅ Installation successful!')"
```

## Step 4: Run First Analysis

### Option A: Quick Python Script

```python
from src.data.loaders import load_wyscout_data
from src.data.extractors import extract_set_pieces, calculate_success_metrics

# Load World Cup 2022 data
data = load_wyscout_data()
print(f"Loaded {len(data)} events")

# Extract set pieces
set_pieces = extract_set_pieces(data)
print(f"Found {len(set_pieces)} set pieces")

# Calculate success metrics
metrics = calculate_success_metrics(set_pieces)
print(f"Goal Rate: {metrics['goal_rate']:.2f}%")
print(f"Success Rate: {metrics['success_rate']:.2f}%")
```

### Option B: Jupyter Notebook

```bash
jupyter notebook notebooks/01_quick_start.ipynb
```

### Option C: Command Line

```bash
python scripts/train_all_models.py
```

## Step 5: Launch Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

This will open the dashboard in your browser at `http://localhost:8501`

---

## 🎉 Done!

You now have a fully working analytics platform!

### What's Next?

1. **Explore the Dashboard** - Try different filters and visualizations
2. **Run Full Analysis** - Check `notebooks/02_full_analysis.ipynb`
3. **Get Tactical Insights** - See `notebooks/03_tactical_insights.ipynb`
4. **Train Custom Models** - Modify `scripts/train_all_models.py`

---

## 📁 Project Structure

```
elite-set-piece-analytics/
├── src/
│   ├── data/           # Data loading and extraction
│   ├── features/       # Feature engineering
│   ├── models/         # ML models
│   └── visualization/  # Plotting functions
├── dashboard/          # Streamlit app
├── notebooks/          # Jupyter notebooks
├── scripts/            # Training scripts
├── tests/              # Unit tests
├── docs/               # Documentation
├── data/               # Data files
└── models/             # Saved models
```

---

## 🆘 Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
cd elite-set-piece-analytics

# Install in development mode
pip install -e .
```

### Missing Data

The platform automatically generates sample World Cup 2022 data if real data files are not present.

### Dashboard Not Loading

```bash
# Try a specific port
streamlit run dashboard/app.py --server.port 8502
```

---

## 📚 Additional Resources

- [Full Documentation](RESULTS.md)
- [API Reference](../src/README.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

**Happy Analyzing! ⚽📊**
