# WUTIS-RNN-VOL

**WUTIS Semester Project SS-25**  
Using Recurrent Neural Networks (RNN) for volatility predictions in financial markets.

---

## Overview

This project leverages RNNs to predict market volatility based on historical data and technical indicators. It includes data fetching, preprocessing, model training, and hyperparameter tuning. The repository is designed to be modular and user-friendly, enabling seamless experimentation and customization.

---

## Features

- **Data Fetching**: Retrieve historical market data using external APIs (e.g., Alpaca API).
- **Technical Indicators**: Generate a variety of technical indicators for enhanced feature engineering.
- **Preprocessing**: Prepare data for CRNN input, including scaling and reshaping.
- **Model Training**: Train a CRNN model for volatility prediction.
- **Hyperparameter Tuning**: Optimize model performance using a configuration file.

---

## How to Use

### 1. Fetch and Preprocess Data
Run the following scripts in sequence to fetch and preprocess the data:

1. **`fetch.py`**  
    Fetch historical market data from external APIs.

2. **`indicators.py`**  
    Generate technical indicators for the fetched data.

3. **`preprocess.py`**  
    Preprocess the data (e.g., scaling, reshaping) to fit the required CRNN input shape.

### 2. Train the Model
Start training the model by running:

```bash
python train.py
```

### 3. Tune the Model
To fine-tune the model, run:

```bash
python -m model.tuning
```

Ensure you have configured the `model_config.yaml` file beforehand.

Rerunning `tuning.py` will reuse the existing Optuna study stored at
`studies/tuning_crnn.sqlite3`. Delete this database file if you want to start a
fresh tuning session.

---

## Configuration

### Alpaca API Configuration
To use the Alpaca API for historical data, define a `config.yaml` file in the `config/` directory as follows:

```yaml
alpaca_api: 
  secret_key: 'your_secret_key'
  api_key: 'your_api_key'
  base_url: 'https://paper-api.alpaca.markets' # Example URL
```

### Model Configuration
Customize the model parameters in `model_config.yaml` to experiment with different architectures and hyperparameters.

---

## Repository Structure

```
wutis-rnn-vol/
├── analysis/             # Data plotting and data analysis
├── config/               # Configuration files
├── data/                 # Data methods
├── model/                # Training and tuning 
├── source/               # Project sources: presentation, documentations etc.
└── README.md             # Project documentation
```

---

## Additional Resources

For more details, visit the [GitHub repository](https://github.com/SukramReburg/wutis-rnn-vol/tree/model).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy experimenting with RNNs for financial volatility predictions!
