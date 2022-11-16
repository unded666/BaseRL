import numpy as np
from copy import deepcopy
from enum import Enum


def get_in_bounds(location: tuple([int, int]), bounds: tuple[int, int]) -> tuple([int, int]):
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

    output_location = list(deepcopy(location))
    for index, (location_i, bounds_i) in enumerate(zip(location, bounds)):
        # deal with values less than 1
        output_location[index] = max(1, location_i)
        # deal with values greater than the boundary
        output_location[index] = min(output_location[index], bounds_i)

    return tuple(output_location)

class TreasureHunt:

    class Direction(Enum):

        UP = 0
        LEFT = 1
        RIGHT = 2
        DOWN = 3

    def __init__(self, field_size: tuple = (5, 5), treasure_location: tuple = (2, 2)):
        """
        creates a treasure hunt object. The treasure is hidden at the treasure location. The grid
        is defined as having coordinates with (1,1) being the top-left corner of the grid, and (X, X)
        being the lower right-hand corner of an X-by-X grid

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

        direction_map = {TreasureHunt.Direction.UP: (0, -1),
                         TreasureHunt.Direction.DOWN: (0, 1),
                         TreasureHunt.Direction.LEFT: (-1, 0),
                         TreasureHunt.Direction.RIGHT: (1, 0)}

        return None

    def reset_prize(self, treasure_location: tuple) -> None:

        return None

    def get_current_state(self) -> int:

        return None

