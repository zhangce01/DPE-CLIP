import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase


template = ['a photo of a {}.',
            'A {} featuring a wide range of color options for easy selection.']

class StanfordCars(DatasetBase):

    dataset_dir = 'stanfordcars'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')
        self.cupl_path = './gpt3_prompts/CuPL_prompts_stanfordcars.json'

        self.template = template

        test = OxfordPets.read_split(self.split_path, self.dataset_dir)

        super().__init__(test=test)