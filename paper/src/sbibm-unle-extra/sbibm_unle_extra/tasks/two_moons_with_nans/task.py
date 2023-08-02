from pathlib import Path
from typing import Optional

import torch
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.two_moons.task import TwoMoons


class TwoMoonsWithNans(TwoMoons):
    def __init__(self):
        """Two Moons"""

        super().__init__()
        # Observation seeds to use when generating ground truth
        self.observation_seeds = [1000011]
        self.num_observations = len(self.observation_seeds)
        self.name = "two_moons_with_nans"
        self.name_display = "Two Moons With Nans"
        self.path = Path(__file__).parent.absolute()

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        two_moons_simulators = super().get_simulator(max_calls).simulator

        def simulator_with_nans(parameters) -> torch.Tensor:
            observations = two_moons_simulators(parameters)
            observations[torch.rand(observations.shape) < 0.5] = torch.nan
            return observations

        return Simulator(task=self, simulator=simulator_with_nans, max_calls=max_calls)
