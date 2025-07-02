# How to run and simulate the model?
In your terminal, here's the template of the command you could enter:
```
python main.py --method [train, predict] --epochs [integer_value] --pooling [0-2] --kernel [1,3,8,16]
```
Example code to train the model:
```
python main.py --method train --epochs 1000 --pooling 1
```
Example code to predict using the model without pooling and with 8 kernels:
```
python main.py --method predict --pooling 0 --kernel 8
```
