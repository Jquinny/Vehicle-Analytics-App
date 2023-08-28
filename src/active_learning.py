from typing import List, Any

import cv2 as cv
import numpy as np
import torch

from sentence_transformers import util


def ENMS(
    classes: List[int],
    confidences: List[float],
    features: torch.tensor,
    similarity_thresh: float = 0.5,
) -> float:
    """computes the image-level entropy as a means of describing the information
    content in the image (and subsequently how useful it will be to train on)

    The algorithm is from https://arxiv.org/abs/2204.07965. It performs entropy-
    based Non-Maxium Suppression in order to find a better representation of the
    information content contained in the image, taking into account instance-level
    redundancy.

    Arguments
    ---------
    classes (List[int]):
        classes corresponding to detected objects in the image
    confidences (List[float]):
        confidences for the detected objects
    features (torch.tensor):
        a len(classes) x n feature matrix where the ith row
        corresponds to the feature vector of the ith instance detected
    similarity_thresh (float):
        the threshold used for filtering similar instances. Any instances
        of the same class with a similarity score above this threshold will be
        filtered out

    Returns
    -------
    float:
        the image-level entropy
    """

    cos_scores = util.cos_sim(features, features)

    E_total = 0
    E_instances = np.array(
        [-conf * np.log2(conf) - (1 - conf) * np.log2(1 - conf) for conf in confidences]
    )
    indicating_set = list(range(len(E_instances)))

    while indicating_set:
        idx = np.argmax(E_instances[indicating_set])
        most_informative_inst = indicating_set[idx]
        indicating_set.pop(idx)
        E_total += E_instances[most_informative_inst]
        for inst in indicating_set[:]:
            if (
                classes[inst] == classes[most_informative_inst]
                and cos_scores[inst, most_informative_inst].item() > similarity_thresh
            ):
                # instances are redundant, get rid of lower information one
                indicating_set.remove(inst)

    return E_total
