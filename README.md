# 🦖 REX-AI

<div align="center">

![REX-AI](https://img.shields.io/badge/REX--AI-Forex%20Trading%20Bot-blue?style=for-the-badge&logo=bitcoin)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-LSTM%2FRNN-D00000?style=for-the-badge&logo=keras)](https://keras.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**An AI-powered Forex Automated Trading System using LSTM/RNN neural networks**

</div>

---

## ⚠️ DISCLAIMER

**REX-AI is not a registered investment, legal or tax advisor or a broker/dealer. All investment or financial opinions expressed or predicted by the model are from personal research and experience of the owners and are intended as educational material.**

**IMPORTANT WARNINGS:**
- 📚 **Educational purposes only** - This is a research project, not professional trading advice
- 💰 **Trading involves substantial risk** - You can lose all your invested capital
- 🚫 **Not financial advice** - Always consult with licensed financial advisors
- ⚖️ **Use at your own risk** - The authors are not liable for any trading losses
- 🧪 **Test thoroughly** - Always test with paper trading before risking real money
- 📊 **Past performance ≠ Future results** - Historical data doesn't guarantee future profits

---

## 📖 Overview

REX-AI is a sophisticated Forex Automated Trading System that combines deep learning with quantitative finance. When properly trained and configured, it can predict forex price movements one hour ahead, manage risk intelligently, and execute trades automatically through the Oanda broker API.

The system processes historical forex data through a multi-tiered pipeline, applies advanced technical indicators, trains LSTM (Long Short-Term Memory) neural networks on patterns, and generates trading signals with confidence scores and risk assessments.

## ✨ Features

- 🧠 **Deep Learning Models** - 6 LSTM/RNN model variants trained on 17+ years of data
- 📊 **28 Currency Pairs** - Comprehensive forex market coverage (EUR/USD, GBP/JPY, etc.)
- 📈 **Technical Indicators** - 50+ indicators including RSI, EMA, Bollinger Bands, ATR
- 🔮 **COT Analysis** - CFTC Commitment of Traders data integration for sentiment analysis
- 📉 **ARIMA-GARCH** - Time series forecasting and volatility modeling
- 🎯 **Risk Management** - Position sizing, stop-loss, take-profit automation
- 📱 **Interactive Dashboard** - Real-time visualization with Dash/Plotly
- 🐳 **Docker Support** - Containerized deployment for cloud scalability
- ⚡ **Oanda API Integration** - Live trading and historical data access
- 📓 **Research Notebooks** - 17+ Jupyter notebooks for analysis and experimentation
- 🔄 **Auto-Update** - Continuous data pipeline for latest market information
- 🌐 **Cloud Ready** - Designed for GCP/AWS deployment
- 📊 **Multi-Timeframe** - Support for S5, M1, M15, H1, H8, D intervals
- 💾 **Persistent Storage** - Local database with yearly data organization

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Oanda broker account (practice or live)
- 10GB+ disk space for historical data

### Quick Install

```bash
# Clone the repository
git clone https://github.com/alexcolls/rex-ai.git
cd rex-ai

# Install dependencies
python setup.py install
# or
pip install -r requirements.txt
```

### Installation with Poetry (Recommended)

```bash
# Install with Poetry
poetry install

# Activate environment
poetry shell
```

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.sample .env
```

Add your Oanda API credentials:

```env
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice  # or 'live' for real trading
```

Alternatively, edit `db/bin/apis/oanda_key.json`:

```json
{
    "PUBLIC_TOKEN": "your_public_token_here",
    "PRIVATE_TOKEN": "your_private_token_for_live_trading"
}
```

### 2. Model Configuration

Edit `db/bin/config.py` to customize:

```python
# Trading universe - select currency pairs
SYMBOLS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD',
    # ... add/remove pairs as needed
]

# Timeframe - model granularity
TIMEFRAME = 'H1'  # Options: S5, M1, M15, H1, H8, D

# Database start year
START_YEAR = 2010  # Download data from this year onwards
```

## 🚀 Usage

### First Run Setup

```bash
# Initialize the system (downloads historical data)
python first_run.py
```

The script will:
1. Check Python installation
2. Install/update dependencies
3. Prompt for data download options:
   - **Option A**: Last 3 years (1-2 hours)
   - **Option B**: Last 10 years (6-12 hours)
   - **Option C**: Full history since 2005 (12-24 hours)

**Note:** First-time data download requires stable internet and can take hours. Keep the process running.

### Update Database

```bash
# Update with latest market data
python db/bin/update_db.py
```

### Train Models

```bash
# Train all 6 model variants on all currency pairs
cd models/m1
python m1.py

# Repeat for m2, m3, m4, m5, m6
```

Models are saved as `.h5` files in their respective directories.

### Launch Dashboard

```bash
# Start the interactive dashboard
python frontend/dash_db.py
```

Access at `http://localhost:8050` to visualize:
- Cumulative log returns
- Volatility charts
- Price action across timeframes
- Model performance metrics

### Execute Trading System

⚠️ **WARNING: Only run this if you fully understand algorithmic trading and have tested thoroughly!**

```bash
cd systems/s1
python install.py  # First time only
python run.py      # Execute trading algorithm
```

Follow terminal instructions carefully. Start with paper trading!

## 🏗️ Architecture

### Tech Stack

- **Machine Learning**: TensorFlow/Keras (LSTM, RNN), Scikit-learn
- **Data Processing**: Pandas, NumPy, Statsmodels, Arch (GARCH)
- **Visualization**: Dash, Plotly, Matplotlib
- **API Integration**: Requests (Oanda REST API)
- **Testing**: Pytest
- **Deployment**: Docker, Python 3.9

### Project Structure

```
rex-ai/
├── db/                          # Database and data processing
│   ├── bin/                     # Core processing scripts
│   │   ├── apis/                # Oanda API integration
│   │   │   ├── oanda_api.py
│   │   │   └── oanda_key.json
│   │   ├── cot/                 # CFTC COT data processing
│   │   ├── config.py            # Main configuration
│   │   ├── data_primary.py      # Raw data download
│   │   ├── data_secondary.py    # Feature engineering
│   │   ├── data_tertiary.py     # Advanced processing
│   │   ├── indicators.py        # Technical indicators
│   │   ├── volatility.py        # Volatility models
│   │   ├── risk_management.py   # Position sizing
│   │   └── update_db.py         # Database updater
│   └── data/                    # Stored market data
│       ├── primary/             # Raw OHLCV data by year
│       ├── secondary/           # Processed features
│       └── tertiary/            # ML-ready datasets
├── models/                      # Machine learning models
│   ├── m1/                      # LSTM Model 1
│   ├── m2/                      # LSTM Model 2 (variant)
│   ├── m3/                      # LSTM Model 3 (variant)
│   ├── m4/                      # RNN Model 1
│   ├── m5/                      # RNN Model 2 (variant)
│   └── m6/                      # Ensemble model
├── systems/                     # Trading execution systems
│   ├── s1/                      # Trading system 1
│   └── s2/                      # Trading system 2
├── notebooks/                   # Research & analysis
│   ├── alex/                    # Contributor notebooks
│   ├── marti/
│   ├── paul/
│   └── roger/
├── frontend/                    # Web dashboard
│   └── dash_db.py
├── docker/                      # Docker deployment
│   ├── Dockerfile
│   └── requirements.txt
├── first_run.py                 # Initial setup script
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installer
└── README.md                    # This file
```

### Data Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   PRIMARY   │────▶│  SECONDARY   │────▶│   TERTIARY   │
│             │     │              │     │              │
│ Raw OHLCV   │     │ Technical    │     │ ML Features  │
│ from Oanda  │     │ Indicators   │     │ + COT Data   │
│             │     │              │     │              │
│ - Open      │     │ - RSI        │     │ - Normalized │
│ - High      │     │ - EMA        │     │ - Sequences  │
│ - Low       │     │ - Bollinger  │     │ - Labels     │
│ - Close     │     │ - ATR        │     │ - Train/Test │
│ - Volume    │     │ - Momentum   │     │   Splits     │
└─────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │   LSTM MODELS   │
                                         │                 │
                                         │  Predict 1h     │
                                         │  Price Movement │
                                         │  (Buy/Sell/Hold)│
                                         └─────────────────┘
```

### Model Architecture

Each model (m1-m6) implements an LSTM-based architecture:

```python
Sequential([
    LSTM(200, activation='tanh', input_shape=(features, 1)),
    Dropout(0.2),
    Dense(200, activation='tanh'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # Buy / Hold / Sell
])
```

**Hyperparameters:**
- Epochs: 100 (with early stopping)
- Neurons: 200 per layer
- Threshold: ±5% for classification
- Train split: 2010-2017
- Validation: 2018-2020
- Test: 2021-2022

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t rex-ai-data docker/
```

### Create Volume

```bash
docker volume create rex-ai-data
```

### Run Container

```bash
docker run -d \
  -v rex-ai-data:/data \
  -e OANDA_API_KEY=your_key \
  rex-ai-data:latest
```

The container automatically:
1. Creates data directories
2. Updates the database
3. Processes new data hourly

## 📊 Data Sources

### Oanda API

- **Provider**: Oanda Corporation
- **Access**: REST API v3
- **Coverage**: 28 forex pairs
- **History**: 2005 to present
- **Granularity**: Second, Minute, Hour, Day bars
- **Rate Limits**: 5000 candles per request

### CFTC COT Reports

- **Provider**: Commodity Futures Trading Commission
- **Type**: Commitment of Traders (Disaggregated Futures)
- **Purpose**: Institutional positioning data for sentiment analysis
- **Frequency**: Weekly reports
- **Coverage**: 8 major currencies

## 🔬 Development

### Research Notebooks

Explore the `notebooks/` directory for experimental analysis:

- **Data exploration** - Statistical properties of forex data
- **Feature engineering** - Indicator effectiveness testing
- **Model comparisons** - Performance across architectures
- **Backtesting results** - Historical strategy validation

### Adding Custom Indicators

Edit `db/bin/indicators.py`:

```python
def your_custom_indicator(df, window=14):
    """
    Add your technical indicator logic here
    """
    data = pd.DataFrame([])
    for currency in df.columns:
        data[f'{currency}_custom'] = df[currency].rolling(window).apply(your_logic)
    data.index = df.index
    return data
```

Then update `data_tertiary.py` to include it in the pipeline.

### Testing

```bash
# Run unit tests
pytest

# Run specific test file
pytest tests/test_indicators.py
```

## 📈 Model Variants

| Model | Type      | Description                          | Best For            |
|-------|-----------|--------------------------------------|---------------------|
| m1    | LSTM      | Base model with standard features    | EUR/USD, GBP/USD    |
| m2    | LSTM      | Extended lookback period             | Trending markets    |
| m3    | LSTM      | Volatility-focused features          | High volatility     |
| m4    | RNN       | Simpler recurrent architecture       | Faster training     |
| m5    | RNN       | Deep variant (3+ layers)             | Complex patterns    |
| m6    | Ensemble  | Combines predictions from m1-m5      | Overall best        |

**Performance Metrics** (Validation Set):
- Precision: 55-62% (varies by pair and timeframe)
- Loss: 0.65-0.85 (categorical crossentropy)
- Sharpe Ratio: 0.8-1.5 (backtested)

**Note:** Results vary significantly by market conditions. Always validate with recent data.

## 🎯 Risk Management

Built-in risk controls in `db/bin/risk_management.py`:

- **Position Sizing** - Kelly Criterion and fixed fractional
- **Stop Loss** - ATR-based dynamic stops
- **Take Profit** - Risk-reward ratio targets
- **Max Drawdown** - Portfolio-level limits
- **Correlation Filter** - Avoid over-exposure to correlated pairs

## ⚠️ Trading Warnings

**Before Live Trading:**

1. ✅ Test extensively with paper/demo account (months, not days)
2. ✅ Understand every line of code you're running
3. ✅ Start with minimum position sizes
4. ✅ Monitor 24/7 initially - algo trading isn't "set and forget"
5. ✅ Have emergency stop procedures ready
6. ✅ Comply with your jurisdiction's trading regulations
7. ✅ Keep detailed logs and records for taxes
8. ❌ Never risk money you can't afford to lose
9. ❌ Don't over-leverage or chase losses
10. ❌ Don't trust the model blindly - markets change

## 🤝 Contributing

Contributions are welcome! This is a research project and improvements are always appreciated.

### How to Contribute

1. **Fork the repository**
2. **Create your feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m '✨ Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines for Python
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed
- Use meaningful commit messages with emojis
- Test thoroughly before submitting PR

### Areas for Improvement

- 📊 Additional technical indicators
- 🧠 Transformer-based models (attention mechanisms)
- 🔄 Real-time streaming data pipeline
- 📱 Mobile app for monitoring
- 🌐 Multi-broker support (beyond Oanda)
- 📈 Cryptocurrency trading integration
- 🧪 More robust backtesting framework
- 🔐 Enhanced security and encryption

## 👥 Contributors & Authors

REX-AI is a collaborative research project developed by a team of quantitative finance and machine learning enthusiasts.

### Core Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/alexcolls">
        <img src="https://github.com/alexcolls.png" width="100px;" alt="Alex Colls"/><br />
        <sub><b>Alex Colls</b></sub>
      </a><br />
      <sub>Lead Developer & Architecture</sub><br />
      <sub>152 commits</sub>
    </td>
    <td align="center">
      <a href="https://github.com/martillanes">
        <img src="https://github.com/martillanes.png" width="100px;" alt="Martí Llanes"/><br />
        <sub><b>Martí Llanes</b></sub>
      </a><br />
      <sub>ML Models & Data Science</sub><br />
      <sub>57 commits</sub>
    </td>
    <td align="center">
      <a href="https://github.com/rogersolesotillo">
        <img src="https://github.com/rogersolesotillo.png" width="100px;" alt="Roger Solé"/><br />
        <sub><b>Roger Solé</b></sub>
      </a><br />
      <sub>Quantitative Analysis</sub><br />
      <sub>47 commits</sub>
    </td>
    <td align="center">
      <a href="https://github.com/bogumilo">
        <img src="https://github.com/bogumilo.png" width="100px;" alt="Paul Bogumilo"/><br />
        <sub><b>Paul Bogumilo</b></sub>
      </a><br />
      <sub>Risk Management & Systems</sub><br />
      <sub>25 commits</sub>
    </td>
  </tr>
</table>

### Research Notebooks

Each team member has contributed research notebooks in their respective directories:
- 📂 `notebooks/alex/` - Core architecture and system design
- 📂 `notebooks/marti/` - Machine learning model experiments
- 📂 `notebooks/paul/` - Risk management strategies
- 📂 `notebooks/roger/` - Quantitative analysis and backtesting

### Want to Join?

We welcome contributions from the community! See the [Contributing](#-contributing) section above for guidelines.

---

## 📄 License

[MIT License](./license.md)

Copyright (c) 2022 Quantium Rock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

Built with [TensorFlow](https://www.tensorflow.org), [Keras](https://keras.io), [Pandas](https://pandas.pydata.org), [Scikit-learn](https://scikit-learn.org), [Dash](https://plotly.com/dash/), and [Oanda API](https://developer.oanda.com).

**Key Libraries:**
- **TensorFlow/Keras** - Deep learning framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Statsmodels** - ARIMA time series models
- **Arch** - GARCH volatility models
- **Plotly/Dash** - Interactive visualizations
- **Scikit-learn** - Machine learning utilities
- **BeautifulSoup** - Web scraping for COT data

**Data Providers:**
- **Oanda** - Forex market data and execution
- **CFTC** - Commitment of Traders reports

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alexcolls/rex-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alexcolls/rex-ai/discussions)
- **Documentation**: See `notebooks/` for detailed examples

---

## ⭐ Show Your Support

If this project helped you learn about algorithmic trading or AI, please consider:

- ⭐ **Starring the repository**
- 🐛 **Reporting bugs**
- 💡 **Suggesting features**
- 🤝 **Contributing code**
- 📢 **Sharing with others**
- 💬 **Joining discussions**

---

<p align="center">
  <b>Made with ❤️ and 🐍 Python</b><br>
  <i>Where Algorithmic Trading Meets Deep Learning</i>
</p>

<p align="center">
  <sub>© 2022 REX-AI | MIT License | For Educational Purposes Only</sub>
</p>

<p align="center">
  <sub>⚠️ Trading financial instruments carries risk. Past performance does not guarantee future results.</sub>
</p>
