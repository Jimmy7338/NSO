"""Run the pinned upstream graph and checkpoint actor on observed occupancy only."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys


def main():
    parent = Path(__file__).resolve().parent
    source = parent / "source/src/scripts"
    protocol = json.loads((parent / "PROTOCOL.json").read_text())
    config = protocol["graph"]
    sys.path.insert(0, str(source))
    reply = sys.stdout

    with redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        import parameter as p
        from agent import Agent
        from model import PolicyNet
        from utils import MapInfo

        p.CELL_SIZE = config["grid_resolution_m"]
        p.SENSOR_RANGE = config["sensor_range_m"]
        p.UTILITY_RANGE = p.SENSOR_RANGE * config["utility_range_factor"]
        p.NODE_RESOLUTION = config["node_spacing_m"]
        p.FRONTIER_CELL_SIZE = p.CELL_SIZE * config["frontier_downsample_factor"]
        p.UPDATING_MAP_SIZE = 4 * p.SENSOR_RANGE + 4 * p.NODE_RESOLUTION
        lattice_spacing = round(p.NODE_RESOLUTION / p.CELL_SIZE)
        assert lattice_spacing * p.CELL_SIZE == p.NODE_RESOLUTION
        pad = lattice_spacing + 1
        net = PolicyNet(p.NODE_INPUT_DIM, p.EMBEDDING_DIM)
        checkpoint = torch.load(source / "model/checkpoint.pth", map_location="cpu", weights_only=True)
        net.load_state_dict(checkpoint["policy_model"])
        net.eval()
        actor = Agent(net)

    anchor = None
    for line in sys.stdin:
        request = json.loads(line)
        with redirect_stdout(sys.stderr), torch.inference_mode():
            row, col = request["cell"]
            if anchor is None:
                anchor = row, col
            row0, col0 = anchor
            belief = np.asarray(request["belief"], dtype=np.int8)
            ros_map = np.where(belief == 1, p.OCCUPIED, belief).astype(np.int8)
            grid = np.pad(ros_map, pad, constant_values=p.UNKNOWN)
            map_info = MapInfo(grid, round(-(col0 + pad) * p.CELL_SIZE, 1),
                               round(-(row0 + pad) * p.CELL_SIZE, 1), p.CELL_SIZE)
            robot = np.round(p.CELL_SIZE * np.array([col - col0, row - row0]), 1)
            actor.node_manager.check_valid_node(robot, map_info)
            if len(actor.node_manager.nodes_dict):
                nearby = actor.node_manager.nodes_dict.nearest_neighbors(robot.tolist(), 1)[0]
                graph_start = nearby.data.coords
            else:
                graph_start = robot
            actor.update_planning_state(map_info, graph_start)
            utility = int(np.sum(actor.key_utility))
            audit = {"graph_nodes": len(actor.key_node_coords), "graph_utility": utility}
            if utility == 0:
                audit["reason"] = "upstream_no_graph_utility"
            elif len(actor.key_neighbor_indices) <= 1:
                audit["reason"] = "upstream_graph_has_no_neighbor"
            else:
                observation = actor.get_observation(robot)
                target, index = actor.select_next_waypoint(observation)
                dx, dy = np.rint(target / p.CELL_SIZE).astype(int)
                audit["target_cell"] = [row0 + int(dy), col0 + int(dx)]
                audit["target_relative_m"] = target.tolist()
                audit["graph_target_index"] = index
        print(json.dumps(audit, allow_nan=False), file=reply, flush=True)


if __name__ == "__main__":
    main()
