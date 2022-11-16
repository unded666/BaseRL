import numpy as np
from enum import Enum

class Treasure_hunt:

    class Direction(Enum):

        UP = 0
        LEFT = 1
        RIGHT = 2
        DOWN = 3

    def __init__(self, field_size: tuple = (5, 5), treasure_location: tuple = (2, 2)):
        """
        creates a treasure hunt object. The treasure is hidden at the treasure location

        :treasure location: the location of the treasure to find
        :field_size: tuple with the maximum value of each dimension
        """

        self.field_dimensions = field_size
        self.treasure = treasure_location
        self.location = (1, 1)
        self.steps_taken = 0

    def take_step(self, action: Direction) -> int:
        """
        takes a step based on a

        """


        return None

    def reset_prize(self, treasure_location: tuple) -> None:

        return None

    def get_current_state(self) -> int:

        return None

