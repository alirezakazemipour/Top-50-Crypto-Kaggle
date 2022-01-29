import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crypto_name", type=str, default="Ethereum")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123, help="randomness seed")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--epoch", type=int, default=30, help="number of epochs")

    return parser.parse_args()
