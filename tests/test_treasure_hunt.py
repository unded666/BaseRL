from unittest import TestCase
from Environments.THunt import TreasureHunt, get_in_bounds
import numpy as np


class TreasureTest(TestCase):

    def setUp(self) -> None:
        """
        creates a default treasure hunt object, with a 6x6 grid and the treasure
        located in location 3,3
        """
        self.hunting_ground = TreasureHunt(field_size=(6, 6),
                                           treasure_location=(3, 3),
                                           )

    def test_get_in_bounds(self) -> None:
        """
        tests whether get_in_bounds correctly leaves a valid location in peace
        tests whether get_in_bounds correctly lowers an X value from out of bounds
        tests whether get_in_bounds correctly lowers a Y value from out of bounds
        tests whether get_in_bounds correctly raises an X value from out of bounds
        tests whether get_in_bounds correctly raises a Y value from out of bounds
        """

        bounds = (5, 5)
        valid_location = (3, 4)
        test_location = get_in_bounds(valid_location, bounds)
        self.assertCountEqual(valid_location, test_location, f"valid location incorrectly modified")
        invalid_x_low = (-3, 4)
        test_location = get_in_bounds(invalid_x_low, bounds)
        self.assertCountEqual(test_location, (1, 4),
                              f"invalid low X not correctly raised, returned location {test_location}")
        invalid_x_high = (7, 4)
        test_location = get_in_bounds(invalid_x_high, bounds)
        self.assertCountEqual(test_location, (5, 4),
                              f"invalid high X not correctly lowered, returned location {test_location}")
        invalid_y_low = (3, 0)
        test_location = get_in_bounds(invalid_y_low, bounds)
        self.assertCountEqual(test_location, (3, 1),
                              f"invalid low Y not correctly raised, returned location {test_location}")
        invalid_y_high = (3, 22)
        test_location = get_in_bounds(invalid_y_high, bounds)
        self.assertCountEqual(test_location, (3, 5),
                              f"invalid high Y not correctly lowered, returned location {test_location}")

    def test_initialisation(self) -> None:
        """
        tests that the steps taken are initialised to zero
        tests that the field size is correct
        tests that the treasure is located in the correct location
        tests that the starting location of the agent is correct
        """

        self.assertEqual(self.hunting_ground.steps_taken, 0,
                         f"initialised steps taken should be zero, but {self.hunting_ground.steps_taken} instead")
        self.assertSequenceEqual((6, 6), self.hunting_ground.field_dimensions,
                                 f"field dimensions incorrectly initialised, read in as "
                                 f"{self.hunting_ground.field_dimensions}")
        self.assertSequenceEqual((3, 3), self.hunting_ground.treasure,
                                 f"treasure misplaced, found in location {self.hunting_ground.treasure}")
        self.assertSequenceEqual((1, 1), self.hunting_ground.location,
                                 f"starting square incorrect, found in location {self.hunting_ground.location}")

    def test_take_step(self) -> None:
        """
        tests that UP moves up correctly
        tests that DOWN moves right correctly
        tests that LEFT moves down correctly
        tests that RIGHT moves left correctly
        tests that a move into a boundary does not move
        tests that a value outside of the correct values does not move
        tests that taking a step returns the correct value
        """

        reward, done, location = self.hunting_ground.take_step(self.hunting_ground.Direction.RIGHT)
        self.assertFalse(done, f"episode returned as finished when incomplete")
        self.assertEqual(reward, -1, f"step counter incorrect")
        self.assertSequenceEqual((2, 1), location, "Right move taken incorrectly")
        _, _, location = self.hunting_ground.take_step(self.hunting_ground.Direction.DOWN)
        self.assertSequenceEqual((2, 2), location, f"Down move taken incorrectly")
        _, _, location = self.hunting_ground.take_step(self.hunting_ground.Direction.LEFT)
        self.assertSequenceEqual((1, 2), location, f"Left move taken incorrectly")
        _, _, location = self.hunting_ground.take_step(self.hunting_ground.Direction.UP)
        self.assertSequenceEqual((1, 1), location, f"Up move taken incorrectly")
        reward, _, location = self.hunting_ground.take_step(self.hunting_ground.Direction.UP)
        self.assertSequenceEqual((1, 1), location, f"Boundary move not treated correctly")
        self.assertEqual(reward, -5, f"step counter incorrectly ignored when bouncing off a boundary")
        _, _, _ = self.hunting_ground.take_step(self.hunting_ground.Direction.RIGHT)
        _, _, _ = self.hunting_ground.take_step(self.hunting_ground.Direction.RIGHT)
        _, _, _ = self.hunting_ground.take_step(self.hunting_ground.Direction.DOWN)
        reward, done, _ = self.hunting_ground.take_step(self.hunting_ground.Direction.DOWN)
        self.assertEqual(reward, 91, f"treasure reward incorrectly returned")
        self.assertTrue(done, f"finished episode incorrectly reported")

    def test_reset(self) -> None:
        """
        Tests that reset puts the agent back in the starting square (1,1)
        Tests that reset places the treasure in the correct specified location
        Tests that reset places the treasure in a random location correctly
        """

        _, _, _ = self.hunting_ground.take_step(self.hunting_ground.Direction.DOWN)
        self.hunting_ground.reset((4, 4))
        self.assertSequenceEqual((1, 1), self.hunting_ground.location, f"Agent location incorrectly reset")
        self.assertSequenceEqual((4, 4), self.hunting_ground.treasure, f"Specified treasure incorrectly placed")
        np.random.seed(42)
        self.hunting_ground.reset()
        self.assertSequenceEqual((4, 5), self.hunting_ground.treasure,
                                 f"reset improperly randomising treasure placement")


    def test_get_current_state(self) -> None:
        """
        tests that the correct initial state is returned
        tests that a mid-play state is correctly returned
        """

        self.assertSequenceEqual((1, 1), self.hunting_ground.location, f"Incorrect state returned")
        for _ in range (3):
            _, _, _ = self.hunting_ground.take_step(self.hunting_ground.Direction.RIGHT)
        self.assertSequenceEqual((4, 1), self.hunting_ground.location,
                                 f"Incorrect dtate returned")


