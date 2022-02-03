import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crypto_name", type=str, default="Ethereum")
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123, help="randomness seed")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epoch", type=int, default=30, help="number of epochs")
    parser.add_argument("--online_wandb", action="store_true", help="whether to connect to WandB cloud")

    return parser.parse_args()
