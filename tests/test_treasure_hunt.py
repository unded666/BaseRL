from unittest import TestCase
from Environments.THunt import treasure_hunt
import numpy as np

class TreasureTest(TestCase):
    def SetUp(self) -> None:

        self.hunting_ground = treasure_hunt(field_size=(6, 6),
                                            treasure_location=(3, 3),
                                            )

    def test_initialisation(self) -> None:

        pass

    def test_take_step(self) -> None:

        pass

    def test_reset_prize(self) -> None:

        pass

    def test_get_current_state(self) -> None:

        pass


