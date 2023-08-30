"""The active learning algorithm is based on this paper https://arxiv.org/abs/2204.07965"""

from collections import defaultdict
from pathlib import Path
from typing import List, Any, Dict

import cv2 as cv
import numpy as np
import torch
import yaml

from PIL import Image
from norfair import Detection
from sentence_transformers import util, SentenceTransformer
from src.utils.image import extract_objects
from src.utils.geometry import points_to_rect


def write_yolo_yaml(abs_img_path: str, cls_map: Dict[int, str]):
    yaml_info = {
        "names": cls_map,
    }
    yaml_path = Path(abs_img_path).parents[1] / "data.yaml"
    yaml_path = yaml_path.resolve()

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_info, f)


def write_yolo_annotation_files(
    detections: List[Detection],
    cls_map: Dict[int, str],
    img: np.ndarray,
    abs_img_path: str,
):
    """helper function for writing images and annotations in yolo format to a .txt file

    Arguments
    ---------
    detections (List[Detection]):
        list of norfair detections containing class and bounding box information
    img (np.ndarray):
        the image to be written to a file
    abs_file_path (str):
        the absolute path to the text file that the annotations will be written to
    """

    # write the image first
    cv.imwrite(abs_img_path, img)

    # write the yaml for class num to name mapping
    write_yolo_yaml(abs_img_path, cls_map)

    # write annotation file
    ann_dir = Path(abs_img_path).parents[1] / "labels"
    if not ann_dir.exists():
        ann_dir.mkdir()

    ann_filename = Path(abs_img_path).stem + ".txt"
    abs_ann_path = ann_dir / ann_filename

    lines = []
    for det in detections:
        cls_num = det.data.get("class")
        rect = points_to_rect(det.points)
        x, y, w, h = rect.to_yolo(img.shape[1], img.shape[0])

        lines.append(f"{cls_num} {x} {y} {w} {h}")
    annotations = "\n".join(lines)
    with open(abs_ann_path, "w") as f:
        f.write(annotations)

    return


def entropy(confidences: List[float]):
    """computes entropy of each instance based on its confidence value"""
    return np.array(
        [-conf * np.log2(conf) - (1 - conf) * np.log2(1 - conf) for conf in confidences]
    )


def ENMS(
    classes: List[int],
    confidences: List[float],
    features: torch.tensor,
    similarity_thresh: float = 0.5,
) -> float:
    """computes the image-level entropy as a means of describing the information
    content in the image (and subsequently how useful it will be to train on)

    Performs entropy-based Non-Maximum Suppression in order to find a better
    representation of the information content contained in the image, taking
    into account instance-level redundancy.

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
        the value that intra-image intra-class similarity must be above in order
        to reject a detected instance during ENMS

    Returns
    -------
    float:
        the image-level entropy
    """

    cos_scores = util.cos_sim(features, features)

    E_total = 0
    E_instances = entropy(confidences=confidences)
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


def intra_class_diversity(
    acquired_image_info: Dict[int, Dict[str, Any]], current_image_info: Dict[str, Any]
) -> float:
    """helper function for doing intra-class diversity reduction across images

    Compares the prototypes of each class found in the current image to the
    prototypes for those classes in each of the acquired images. If the similarity
    is too high, reject the image as it would provide redundant information

    Arguments
    ---------
    acquired_image_info (Dict[int, Dict[str, Any]]):
        the image info for the images acquired so far through diverse prototype
    current_image_info (Dict[str, Any]):
        the image info for the current image being processed in diverse prototype

    Returns
    -------
    float:
        the minimum of the max similarities computed across the different class
        prototypes
    """
    if not acquired_image_info:
        # no images acquired yet, just return 0
        return 0

    # gather up prototypes from acquired images, group them together by class
    acquired_protos: Dict[int, List[np.ndarray]] = defaultdict(lambda: [])
    for image_info in acquired_image_info.values():
        for cls_num, prototype in image_info.get("prototypes").items():
            acquired_protos[cls_num].append(prototype)

    # stack prototype lists for efficient cosine similarity calculation
    for cls_num, proto_list in acquired_protos.items():
        acquired_protos[cls_num] = np.stack(proto_list)

    current_protos = current_image_info.get("prototypes")
    min_all_sim = np.inf
    for cls_num in current_protos.keys():
        if acquired_protos.get(cls_num, None) is None:
            # no prototypes made for this class yet
            continue

        # compute cosine similarities for all prototypes of this class
        similarities = util.cos_sim(acquired_protos[cls_num], current_protos[cls_num])
        max_sim = torch.max(similarities)
        if max_sim < min_all_sim:
            min_all_sim = max_sim

    if min_all_sim == np.inf:
        # there was no prototypes to compare too meaning this is the first one
        return 0
    else:
        return min_all_sim.item()


def minority_class_probabilities(
    minority_classes: List[int], detections: List[Detection]
) -> Dict[int, float]:
    """helper for computing the max probability for each minority class in the image

    This is used during inter-class balancing, giving the max probability of
    occurrence for each minority class to be used for further calculation

    Arguments
    ---------
    minority_classes (List[int]):
        list of the minority object classes
    detections (List[Detection]):
        list of norfair detection objects containing the necessary class and
        probability information

    Returns
    -------
    Dict[int, float]:
        the buckets corresponding to each minority class that hold the max
        probability found for that class
    """
    probability_buckets: Dict[int, float] = {cls: 0 for cls in minority_classes}
    for det in detections:
        cls_num = det.data.get("class")
        conf = det.data.get("conf")
        if cls_num in minority_classes and conf > probability_buckets[cls_num]:
            probability_buckets[cls_num] = conf

    return probability_buckets


def diverse_prototype(
    candidate_images: Dict[int, Dict[str, Any]],
    minority_classes: List[int],
    quota_buckets: Dict[int, float],
    budget: int,
    intra_thresh: float = 0.85,  # clip computes very similar image embeddings
    inter_thresh: float = 0.3,
):
    """Implementation of the diverse prototype algorithm from the paper linked
    at the top of the file.

    Computes best images from the sorted candidate images based on their intra-class
    diversity across images and inter-class balancing across images.

    NOTE: candidate images are expected to be sorted from highest image entropy
    to lowest image entropy.

    Arguments
    ---------
    candidate_images (Dict[int, Dict[str, Any]]):
        the sorted images containing all detection and prototype information
        necessary to complete the algorithm
    minority_classes (List[int]):
        the list of minority classes for active learning
    quota_buckets (Dict[int, float]):
        the quota buckets corresponding to the minority classes. These are used
        to ensure we get a fair share of minority classes
    budget (int):
        the max number of images we can acquire during this active learning
        acquisition cycle
    intra_thresh (float):
        the threshold for performing intra-class reduction
    inter_thresh (float):
        the threshold for performing inter-class reduction

    Returns
    -------
    Dict[int, Dict[str, Any]]:
        the image info dictionary mapping frame indexes to the information
        necessary to build up an active learned dataset
    """
    acquired_images: Dict[int, Dict[str, Any]] = {}
    for idx, (frame_idx, image_info) in enumerate(candidate_images[:]):
        if not minority_classes:
            # we have exhausted all minority quotas, fill in the rest of the budget
            # with highest entropy images remaining
            break

        if not intra_class_diversity(acquired_images, image_info) < intra_thresh:
            # didn't pass intra-class reduction test
            continue

        minority_prob_buckets = minority_class_probabilities(
            minority_classes, image_info.get("detections")
        )
        if not max(minority_prob_buckets.values()) > inter_thresh:
            # didn't pass inter-class reduction test
            continue

        # passed both checks, acquire the frame and remove from sorted_images in case
        # we need to go back through the rest afterwards
        acquired_images[frame_idx] = image_info
        candidate_images.pop(idx)
        for cls in minority_classes[:]:
            # update quotas
            if minority_prob_buckets[cls] > inter_thresh:
                # likely an instance from this class in the image
                quota_buckets[cls] -= 1
                if quota_buckets[cls] < 0:
                    minority_classes.remove(cls)

    # just in case there weren't enough frames with minority classes, fill up until full
    while len(acquired_images) < budget:
        if len(candidate_images) == 0:
            # completely out of images, break
            break

        frame_idx, image_info = candidate_images[0]  # since it was sorted in descending
        acquired_images[frame_idx] = image_info
        candidate_images.pop(0)

    return acquired_images


def active_learn(
    image_info: Dict[int, Dict[str, Any]],
    all_classes: List[int],
    minority_classes: List[int],
    budget: int,
    similarity_thresh: float = 0.5,
    intra_thresh: float = 0.85,
    inter_thresh: float = 0.3,
) -> Dict[int, Dict[str, Any]]:
    """completes the active learning acquisition cycle. Completes necessary
    setup for the ENMS and Diverse Prototype algorithms, and then runs them
    to find the best images for acquisition.

    NOTE: the format for the image_info dict originally is frame indexes for keys, and then
    the value dictionary is of the form:
    {
        "detections": List[norfair.tracking.Detection]
    }

    The value dictionary should only contain the detections key upon entry into
    the function, and it will be updated in place to hold the entropy and
    prototypes in the form:
    {
        "detections": List[norfair.tracking.Detection],
        "entropy": float,
        "prototypes": Dict[int, np.ndarray],
    }

    The entropy per image is computed using Entropy-Based Non-Maximum Suppression

    The prototype dictionary has the class num as keys and the prototype embedding
    for that class as the value. The prototypes are computed according to equation
    3 in the paper linked at the top of the file

    Arguments
    ---------
    image_info (Dict[int, Dict[str, Any]]):
        dictionary containing a mapping of frame index to image information
    all_classes (List[int]):
        the list of all the classes that could have been detected
    minority_classes (List[int]):
        the list of all the minority classes to focus on during active learning
    budget (int):
        the max number of images that can be acquired during this image
        acquisition cycle
    similarity_thresh (float):
        the value that intra-image intra-class similarity must be above in order
        to reject a detected instance during entropy calculation
    intra_thresh (float):
        the threshold for performing intra-class reduction
    inter_thresh (float):
        the threshold for performing inter-class reduction

    Returns
    -------
    Dict[int, Dict[str, Any]]:
        the keys are frame indexes, and the values are image information
        dictionaries holding the necessary detection objects for writing out
        images to directories in the proper format
    """

    # load CLIP model for computing image vector embeddings
    model = SentenceTransformer("clip-ViT-B-32")

    for frame_idx, info in image_info.items():
        detections = info.get("detections")
        classes = np.array([det.data.get("class") for det in detections])
        confidences = np.array([det.data.get("conf") for det in detections])
        single_object_imgs = extract_objects(detections)

        embeddings = model.encode(
            [Image.fromarray(img) for img in single_object_imgs],
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # delete all stored images inside of the detection objects (save memory)
        for det in detections:
            if det.data.get("img") is not None:
                del det.data["img"]

        img_entropy = ENMS(
            classes, confidences, embeddings, similarity_thresh=similarity_thresh
        )

        # prototype calculation
        prototypes: Dict[int, np.ndarray] = {}
        for cls_num in set(classes):
            instance_entropies = entropy(confidences[classes == cls_num])
            instance_embeddings = embeddings[classes == cls_num, :]
            weighted_embeddings = (
                np.expand_dims(instance_entropies, 1) * instance_embeddings
            )
            weighted_embeddings_sum = np.sum(weighted_embeddings, 0)
            prototypes[cls_num] = (
                1 / np.sum(instance_entropies)
            ) * weighted_embeddings_sum

        image_info[frame_idx].update(
            {
                "entropy": img_entropy,
                "prototypes": prototypes,
            }
        )

    # sort the images according to entropy
    sorted_images = sorted(
        image_info.items(), key=lambda item: item[1].get("entropy", 0), reverse=True
    )

    # minority class budget calculation
    minority_num = len(minority_classes) if minority_classes else 1
    alpha = minority_num / len(all_classes)
    beta = alpha + (1 - alpha) / 2 if alpha < 1 else 1
    quota_value = (beta / minority_num) * budget
    quota_buckets = {cls_num: quota_value for cls_num in minority_classes}

    return diverse_prototype(
        candidate_images=sorted_images,
        minority_classes=minority_classes,
        quota_buckets=quota_buckets,
        budget=budget,
        intra_thresh=intra_thresh,
        inter_thresh=inter_thresh,
    )
