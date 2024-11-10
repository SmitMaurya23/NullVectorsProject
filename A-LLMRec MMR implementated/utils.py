import os
from datetime import datetime
from pytz import timezone
import numpy as np

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths



###########################################################
#########################New ADDITION################################################
def mmr_rerank(recommendations, embeddings, relevance_scores, lambda_param=0.5, top_n=10):
    """
    Apply MMR to diversify recommendations.
    
    :param recommendations: List of items sorted by relevance score.
    :param embeddings: Embeddings for each recommended item.
    :param relevance_scores: Dictionary of relevance scores for each item.
    :param lambda_param: Balancing factor between relevance and diversity (0 = all diversity, 1 = all relevance).
    :param top_n: Number of final recommendations to return.
    :return: List of re-ranked recommendations.
    """
    selected = []
    while len(selected) < top_n and recommendations:
        remaining = list(set(recommendations) - set(selected))

        if not remaining:  # Check to prevent empty sequence errors
            break

        mmr_scores = [
            lambda_param * relevance_scores[item] -
            (1 - lambda_param) * max([np.dot(embeddings[item].cpu().numpy(), embeddings[s].cpu().numpy()) for s in selected] or [0])
            for item in remaining
        ]

        # Avoid errors from empty mmr_scores list
        if not mmr_scores:
            break

        # Select item with max MMR score and add to selected
        selected_item = remaining[np.argmax(mmr_scores)]
        selected.append(selected_item)

    return selected
###########################################################
#########################New ADDITION################################################
    
    