# phishing-detector-model

A basic machine learning project that detects whether a URL is phishing or not.

## Files

* `train.py` – trains the model and saves it
* `predict.py` – loads the model and makes predictions
* `requirements.txt` – dependencies
* `phishing-detector-model/` – saved model files

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/tmestery/phishing-detector-model.git
cd phishing-detector-model
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

This trains the model and saves it to the `phishing-detector-model/` folder.

## Predict

```bash
python predict.py
```

Edit `predict.py` to test your own URLs.

## Author

@tmestery
