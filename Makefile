.PHONY: all setup train-keras train-torch clean

PY=python

setup:
	$(PY) -m pip install -r requirements.txt

train-keras:
	$(PY) scripts/train_mnist_keras.py

train-torch:
	$(PY) scripts/train_mnist_torch.py

clean:
	rm -rf outputs/*.png outputs/*.csv outputs/*.txt outputs/*.json
