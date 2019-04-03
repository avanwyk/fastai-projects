""" Antelope Classification using FastAI

This script is an end-to-end case study of creating a custom image dataset of
major African antelope and training a deep convolutional neural network to
classify each species.

The basic workflow is as follows:
1. Download images of each antelope and build a dataset.
2. Pre-process and prepare the dataset for learning.
3. Create a deep neural network model for classification.
4. Train the DNN using transfer learning on the data.
5. Validate and evaluate the model.

"""
import logging
from typing import List

from fastai.vision import *
from fastai.metrics import error_rate
from google_images_download import google_images_download

logging.basicConfig(level=logging.INFO)


ANTELOPE = ['kudu', 'eland', 'sable antelope', 'roan antelope', 'waterbuck',
            'impala antelope', 'nyala', 'bushbuck', 'tsessebe',
            'lichtensteins hartebeest', 'grey duiker', 'steenbok',
            'klipspringer']

DATA_PATH = Path('data')
VALID_PCT = 0.2
IMAGE_SIZE = 224
BATCH_SIZE = 32
ARCHITECTURE = models.resnet34

def download_antelope_images(output_path: Path, limit: int = 50) -> None:
    """Download images for each of the antelope to the output path.
    
    Each species is put in a separate sub-directory under output_path.
    """
    try:
        if len(output_path.ls()) > 0:
            logging.info(f"Directory '{output_path}' is not empty. Skipping image download.")
            return
    except FileNotFoundError:
        logging.info(f"Directory '{output_path} does not exist and will be created.")

    
    response = google_images_download.googleimagesdownload()

    for antelope in ANTELOPE:
        for gender in ['male', 'female']:
            output_directory = str(output_path/antelope).replace(' ', '_')

            arguments = {
                'keywords': f'wild {antelope} {gender} -hunting -stock',
                'output_directory': output_directory,
                'usage_rights': 'labeled-for-nocommercial-reuse',
                'no_directory': True,
                'size': 'medium',
                'limit': limit
            }
            response.download(arguments)


def train_model(data_path: Path, valid_pct, image_size, batch_size, architecture) -> Learner:
    """Train a deep convolutional NN classifier on the downloaded data.
    """
    image_data = ImageDataBunch.from_folder(data_path, valid_pct=valid_pct,\
                                            ds_tfms=get_transforms(), size=image_size,
                                            bs=batch_size).normalize(imagenet_stats)

    learner = cnn_learner(image_data, architecture, metrics=error_rate)
    learner.fit_one_cycle(4, max_lr=slice(1e-3, 1e-2))
    learner.unfreeze()
    learner.fit_one_cycle(4, max_lr=slice(1e-6, 1e-4))
    return learner

if __name__ == '__main__':
    download_antelope_images(DATA_PATH)
    
    learner = train_model(DATA_PATH, VALID_PCT, IMAGE_SIZE, BATCH_SIZE, ARCHITECTURE)

    print(f'Error rate: {learner.recorder.metrics[-1]}')
    