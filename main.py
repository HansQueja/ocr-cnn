import argparse

from helpers.train import train
from helpers.predict import predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="tells the program if train or predict", type=str, default="train")
    parser.add_argument("--epochs", help="total number of epochs", type=int, default=1000)
    parser.add_argument("--pooling", help="if mean, max, or no pooling", type=int, default=0)
    parser.add_argument("--kernel", help="number of kernels trained", type=int, default=3)
    args = parser.parse_args()

    if args.method == "train":
        train(args.epochs, args.pooling)
    else:
        predict(args.pooling, args.kernel)

if __name__ == "__main__":
    main()