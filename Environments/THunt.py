import numpy as np
from enum import Enum

def get_in_bounds (location: tuple([int, int]), bounds: tuple[int, int]) -> tuple([int, int]):
    """
    determines if a location is within 1 and the upper bounds given by bounds.
    If the value is outside of those bounds, it is moved the shortest distance
    to get into those bounds. The new location is then returned

    Parameters:
        location: the desired new coordinates
        bounds: the coordinate boundaries

    Returns:
        new location: the modified (if necessary) new location
    """

    return None

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

    def take_step(self, action: Direction) -> (int, bool, tuple[int, int]):
        """
        takes a step in the specified direction. increments the number of steps taken
        and returns the prize amount

        Parameters:
            action: which direction to move the location

        Returns:
            reward value: 100 if the prize found, 0 elsewise, minus a penalty for steps taken
            done: boolean value showing the location
            state: location of the agent
        """

        return None

    def reset_prize(self, treasure_location: tuple) -> None:

        return None

    def get_current_state(self) -> int:

        return None

