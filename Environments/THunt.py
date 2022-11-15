import numpy as np

class treasure_hunt:

    def __init__(self, treasure_location: tuple = (2, 2)):
        """
        creates a treasure hunt object. The treasure is hidden at the treasure location

        :treasure location: the location of the treasure to find
        """

        self.treasure = treasure_location
        self.location = (1,1)

    def take_step(self, action: int) -> int:

        return None

    def reset_prize(self, treasure_location: tuple) -> None:

        return None

    def get_current_state(self) -> int:

        return None

