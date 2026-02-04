# Stock Price Prediction with Custom Loss

This project predicts future OHLC prices using a deep learning model with a custom loss function
that reflects both price error and directional consistency.

## Motivation
Standard MSE-based training focuses only on numeric error.  
However, in financial forecasting, predicting the correct direction of price movement is also important.  
Therefore, a custom loss combining price MSE and directional loss was designed.

## Method
- Multi-input model (Price, Volume, Stock ID)
- CNN + BiLSTM architecture
- Stock ID embedding
- Custom loss: Price MSE + Direction Loss

## Dataset
- Source: Yahoo Finance  
- Period (Train): 2020-01-01 ~ 2024-12-31  
- Period (Test): 2025-01-01 ~

## Environment
- Python 3.10
- TensorFlow 2.10.0
- NumPy 1.26.4

## Result
- Price prediction accuracy: 98.05%
- Selected as presenter for final project

## Future Work
- Model architecture optimization
- Additional technical indicators

## How to Run
```bash
pip install -r requirements.txt
python train.py
python test.py




