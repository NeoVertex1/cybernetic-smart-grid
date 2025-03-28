import numpy as np
import pandas as pd
import h5py
import json
import pickle
import logging
from typing import List, Dict, Any
from datetime import datetime

# Constants inspired by Galileo-Tensor Solution
PERFECT_FIFTH_STABILITY = 0.9999206  # Harmonic stabilization factor
COHERENCE_TIME = 50.0  # Reduced coherence time for stronger damping
FLOW_LIMIT = 50.0  # Strict power flow limit (W)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"smartgrid_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Node:
    """Class representing a node in the smart grid."""
    def __init__(self, name: str, imbalance: float, is_fixed: bool = False):
        self.name: str = name
        self.imbalance: float = imbalance  # Power imbalance in watts (W)
        self.net_flow: float = 0.0  # Net power flow in watts (W)
        self.is_fixed: bool = is_fixed  # Whether the node can exchange power

    def update_imbalance(self, power_flow: float) -> None:
        """Update the node's imbalance and net flow."""
        if not self.is_fixed:
            self.imbalance += power_flow
            self.net_flow = power_flow

class Controller:
    """Proportional controller for smart grid power flow management."""
    def __init__(self, k: float = 0.05):
        self.k: float = k  # Proportional gain (reduced for stability)
        self.iteration: int = 0

    def compute_dynamic_targets(self, nodes: List[Node]) -> tuple[float, float]:
        """Compute dynamic targets for adjustable nodes based on total system power."""
        total_power = sum(node.imbalance for node in nodes)
        fixed_power = sum(node.imbalance for node in nodes if node.is_fixed)
        adjustable_nodes = sum(1 for node in nodes if not node.is_fixed)

        if adjustable_nodes == 0:
            return 0.0, 0.0

        # Target for adjustable nodes (Node 1 and Node 2)
        adjustable_power = total_power - fixed_power
        target = adjustable_power / adjustable_nodes  # Equal distribution
        return target, target

    def compute_power_flow(self, node1: Node, node2: Node) -> float:
        """Compute power flow using proportional control with harmonic stabilization."""
        # Compute dynamic targets
        target1, target2 = self.compute_dynamic_targets([node1, node2])

        # Proportional control
        p1 = self.k * (target1 - node1.imbalance)
        p2 = self.k * (target2 - node2.imbalance)
        power_flow = (p1 - p2) / 2  # Positive: Node 1 to Node 2, Negative: Node 2 to Node 1

        # Apply harmonic stabilization and strong damping
        decay_factor = np.exp(-self.iteration / COHERENCE_TIME)
        stabilized_flow = power_flow * PERFECT_FIFTH_STABILITY * decay_factor

        # Apply strict flow limit
        stabilized_flow = min(max(stabilized_flow, -FLOW_LIMIT), FLOW_LIMIT)

        self.iteration += 1
        return stabilized_flow

class SmartGrid:
    """Smart grid simulation with proportional control and harmonic stabilization."""
    def __init__(self):
        self.nodes: List[Node] = [
            Node("Node 1", -600.0, is_fixed=False),
            Node("Node 2", -400.0, is_fixed=False),
            Node("Node 3", -800.0, is_fixed=True),
            Node("Node 4", -800.0, is_fixed=True),
        ]
        self.controller: Controller = Controller()

    def compute_total_imbalance(self) -> float:
        """Compute the total imbalance of the system."""
        return sum(abs(node.imbalance) for node in self.nodes)

    def simulate(self, num_iterations: int) -> List[Dict[str, Any]]:
        """Run the smart grid simulation for the specified number of iterations."""
        logger.info(f"Starting simulation with {num_iterations} iterations")
        simulation_data = []

        for i in range(num_iterations):
            # Compute power flow between Node 1 and Node 2
            node1, node2 = self.nodes[0], self.nodes[1]
            power_flow = self.controller.compute_power_flow(node1, node2)

            # Update imbalances
            node1.update_imbalance(-power_flow)  # Node 1 loses power if flow is positive
            node2.update_imbalance(power_flow)   # Node 2 gains power if flow is positive

            # Compute total imbalance
            total_imbalance = self.compute_total_imbalance()

            # Collect data for this iteration
            iteration_data = {
                "iteration": i + 1,
                "node1_imbalance": node1.imbalance,
                "node2_imbalance": node2.imbalance,
                "node3_imbalance": self.nodes[2].imbalance,
                "node4_imbalance": self.nodes[3].imbalance,
                "node1_net_flow": node1.net_flow,
                "node2_net_flow": node2.net_flow,
                "node3_net_flow": self.nodes[2].net_flow,
                "node4_net_flow": self.nodes[3].net_flow,
                "power_flow": power_flow,
                "total_imbalance": total_imbalance
            }
            simulation_data.append(iteration_data)

            # Log the iteration
            logger.info(f"Iteration {i + 1}")
            for node in self.nodes:
                logger.info(f"{node.name} imbalance: {node.imbalance:.2f} W (net_flow={node.net_flow:.2f} W)")
            logger.info(f"Total imbalance: {total_imbalance:.2f} W")
            logger.info(f"Power flow: {power_flow:.2f} W ({'Node 1 to Node 2' if power_flow >= 0 else 'Node 2 to Node 1'})")

        # Save the data
        self.save_data(simulation_data)
        return simulation_data

    def save_data(self, simulation_data: List[Dict[str, Any]]) -> None:
        """Save simulation data in multiple formats."""
        try:
            # Pickle
            with open("smartgrid_data.pkl", "wb") as f:
                pickle.dump(simulation_data, f)
            logger.info("Successfully saved data to smartgrid_data.pkl")
        except Exception as e:
            logger.error(f"Failed to save smartgrid_data.pkl: {e}")

        try:
            # HDF5
            with h5py.File("smartgrid_data.h5", "w") as f:
                for key in simulation_data[0].keys():
                    data = [float(d[key]) for d in simulation_data]
                    f.create_dataset(key, data=data)
            logger.info("Successfully saved data to smartgrid_data.h5")
        except Exception as e:
            logger.error(f"Failed to save smartgrid_data.h5: {e}")

        try:
            # CSV
            df = pd.DataFrame(simulation_data)
            df.to_csv("smartgrid_data.csv", index=False)
            logger.info("Successfully saved data to smartgrid_data.csv")
        except Exception as e:
            logger.error(f"Failed to save smartgrid_data.csv: {e}")

        try:
            # JSON
            with open("smartgrid_data.json", "w") as f:
                json.dump(simulation_data, f, indent=4)
            logger.info("Successfully saved data to smartgrid_data.json")
        except Exception as e:
            logger.error(f"Failed to save smartgrid_data.json: {e}")

def main():
    """Main function to run the smart grid simulation."""
    # Run unit tests
    import unittest
    class TestSmartGrid(unittest.TestCase):
        def setUp(self):
            self.grid = SmartGrid()

        def test_initial_imbalance(self):
            total_imbalance = self.grid.compute_total_imbalance()
            self.assertEqual(total_imbalance, 2600.0, "Initial imbalance should be 2600 W")

    unittest.main(argv=[''], exit=False)

    # Run the simulation
    grid = SmartGrid()
    grid.simulate(num_iterations=50)

if __name__ == "__main__":
    main()
