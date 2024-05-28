from . import sentence_pooling
from . import k_word_pooling
from . import syntax_tree_pooling

import argparse


def main():
    parser = argparse.ArgumentParser(description='Shapley Value Pooling')
    parser.add_argument('--pooling_strategy', type=str, default='sentence', help='sentence or k_word')

    args = parser.parse_args()

    match args.pooling_strategy:
        case "sentence":
            sentence_pooling.main()
        case "k_word":
            k_word_pooling.main()
        case "st":
            syntax_tree_pooling.main()
        case _:
            raise ValueError(f"Unknown/unsupported pooling strategy {args.pooling_strategy}")


if __name__ == "__main__":
    main()
