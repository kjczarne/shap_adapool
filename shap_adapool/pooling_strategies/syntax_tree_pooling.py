import spacy
from spacy import displacy
import pickle
import shap
from pprint import pprint
from typing import List, Any
import numpy as np

from ..types import TokenDtype
from ..initializer import init
from ..pooler import shap_value_pooler, two_element_sum
from ..token_concatenation import token_concat
from ..plotting import save_plot

Node = Any
SpaCySentence = Any


def print_children(token, prefix=""):
    """A simple function demonstrating how to traverse a dependency tree."""
    print(prefix + f"({token.text}, {token.dep_})")
    for child in token.children:
        print_children(child, prefix + "  ")


def build_token_tree_table(trees: List[SpaCySentence]):
    """Builds a table of tokens and their respective levels in the dependency tree.
    For a given list of trees (list of SpaCy sentences), the table will look like this:

    ```python
    [
        [(0, token), (1, token), (2, token), ...],
        [(0, token), (1, token), (2, token), ...],
        ...
    ]
    ```
    """

    current_level = 0
    tree_table = []

    def wrapper(node: Node, current_level: int):
        current_level += 1
        children_and_levels = []

        # Perform postorder tree walk and collect the level indices:
        for child in node.children:
            children_and_levels.extend(wrapper(child, current_level))

        # Append the current node and its tree level index to the list
        children_and_levels.append((current_level, node))
        return children_and_levels

    # Do the walk separately for each tree in the document:
    for tree in trees:
        # Reset the index:
        current_level = 0
        tree_table.append(wrapper(tree.root, current_level))

    return tree_table


def main():

    init()

    # TODO: Replace this with the actual text from the Open Canada dataset
    text = "I don’t want to talk to you no more, you empty-headed animal food trough wiper! I fart in your general direction! Your mother was a hamster and your father smelt of elderberries!"

    # The Transformer pipeline is recommended here as it produces more accurate dependency trees:
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)

    # Demonstrating basic tree traversal:
    for sentence in doc.sents:
        root = sentence.root
        print_children(root)

    # Create an array of random Shapley Values:
    shap_values = np.random.rand(len(doc))

    # Build the token tree table:
    token_tree_table = build_token_tree_table(doc.sents)  

    # Flatten the table:
    token_tree_table_flat = [item for sublist in token_tree_table for item in sublist]

    # Sort the flattened list in order of index in the original document:
    token_tree_table_flat = sorted(token_tree_table_flat, key=lambda x: x[1].i)

    # Split the list into a list of tree levels and a list of tokens:
    tree_levels, tokens = zip(*token_tree_table_flat)

    # Define the end and write conditions for the token concatenation:
    def _end_condition(token: TokenDtype, idx: int, threshold: int) -> bool:
        if tree_levels[idx] <= threshold:
            return True
        return False

    def _write_condition(token: TokenDtype, idx: int, *args, **kwargs) -> bool:
        return True

    # Concatenate the tokens according to the provided conditions:
    # FIXME: missing last part of the document!
    combined, index_map = token_concat(tokens,
                                       write_condition=_write_condition,
                                       end_condition=_end_condition,
                                       add_whitespace=True,
                                       flush_buffer_on_end=True,
                                       threshold=2)

    # Pool the values together aggregating using summation:
    pooled_values = shap_value_pooler(shap_values,
                                      index_map,
                                      two_element_sum)

    # Construct a fake `Explanation` object:
    pooled_shapley_values = np.array(list(pooled_values))
    exp = shap.Explanation(values=pooled_shapley_values[None, :],
                           base_values=np.random.random(pooled_shapley_values.shape[0]),
                           data=(list(combined),))

    # Display the dependency tree (if running in an interactive session):
    # displacy.serve(doc, style="dep")

    # Save the text plot:
    plot = shap.plots.text(exp, display=False)
    name = "syntax_tree"
    save_plot(plot, name + "_pooling")


if __name__ == "__main__":
    main()
