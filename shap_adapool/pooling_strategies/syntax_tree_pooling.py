import spacy
from pprint import pprint


def print_children(token, prefix=""):
    """A simple function demonstrating how to traverse a dependency tree."""
    print(prefix + f"({token.text}, {token.dep_})")
    for child in token.children:
        print_children(child, prefix + "  ")


def main():
    nlp = spacy.load("en_core_web_sm")

    # At the moment we're just showing how to traverse the tree:
    doc = nlp("I don’t want to talk to you no more, you empty-headed animal food trough wiper! I fart in your general direction! Your mother was a hamster and your father smelt of elderberries!")

    for sentence in doc.sents:
        root = sentence.root
        print_children(root)


if __name__ == "__main__":
    main()
