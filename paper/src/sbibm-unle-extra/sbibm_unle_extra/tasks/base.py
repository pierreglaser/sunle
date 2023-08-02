from jax import Array


class SimulatorWithPrecomputedDataset:
    def __init__(self, simulator, precomputed_dataset_loader) -> None:
        self.simulator = simulator
        self.get_large_precomputed_dataset = precomputed_dataset_loader

    def __call__(self, parameters) -> Array:
        return self.simulator(parameters)
