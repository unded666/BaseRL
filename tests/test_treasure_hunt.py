from unittest import TestCase
from Environments.THunt import treasure_hunt
import numpy as np


class TreasureTest(TestCase):

    def setUp(self) -> None:
        """
        creates a default treasure hunt object, with a 6x6 grid and the treasure
        located in location 3,3
        """
        self.hunting_ground = treasure_hunt(field_size=(6, 6),
                                            treasure_location=(3, 3),
                                            )

    def test_initialisation(self) -> None:
        """
        tests that the steps taken are initialised to zero
        tests that the field size is correct
        tests that the treasure is located in the correct location
        tests that the starting location of the agent is correct
        """

        self.assertEqual(self.hunting_ground.steps_taken, 0,
                         f"initialised steps taken should be zero, but {self.hunting_ground.steps_taken} instead")
        self.assertCountEqual((6, 6), self.hunting_ground.field_dimensions,
                              f"field dimensions incorrectly initialised, read in as "
                              f"{self.hunting_ground.field_dimensions}")
        self.assertCountEqual((3, 3), self.hunting_ground.treasure,
                              f"treasure misplaced, found in location {self.hunting_ground.treasure}")
        self.assertCountEqual((1, 1), self.hunting_ground.location,
                              f"starting square incorrect, found in location {self.hunting_ground.location}")


    def test_take_step(self) -> None:

        pass

    def test_reset_prize(self) -> None:

        pass

    def test_get_current_state(self) -> None:

        pass


