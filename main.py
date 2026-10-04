import sys
import time
import math
import random
import threading
import traceback
from collections import deque
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"[*] Engine Online: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

# ---------------------------------------------------------------------------
# Built-in Maze Presets (With Bounce Blocks)
# ---------------------------------------------------------------------------
# '#' = Low Wall (Jumpable if cleared; landing on it = DEATH)
# 'R' = Tall Red Tower (Blocks all flight; contact = DEATH)
# 'B' = Bounce Block (Bounces you high enough to fly OVER Red Towers!)
# '.' = Walkable Floor
# 'S' = Start Position (1, 1)
# 'G' = Goal Position (14, 14)
PRESETS = {
    "Default Chasm": [
        "################",
        "#S....##########",
        "#####.##########",
        "#####.##RRRR####",
        "#####.##....####",
        "#####.##.#######",
        "###...##.#######",  # Jumpable chasm gap
        "###.####.#######",
        "###.####.#######",
        "#..B####....####",  # Row 9: Bounce block at (9, 3) facing Red Barricade!
        "#.RRRRRRRR#.####",  # Deep detour barrier
        "#..........#.###",
        "############.###",
        "############.###",
        "############..G#",  # Safe inside boundary wall
        "################",
    ],
    "Double Chasm": [
        "################",
        "#S....#...R....#",
        "#####.#.#.R.##.#",
        "###...#.#.R.##.#",
        "#.....#.#.R....#",
        "#.RRRR#.#.RRRR.#",
        "#.#...#.#......#",
        "#.#.#.######.#.#",
        "#...#.######.#.#",
        "###.R...##...#.#",
        "###.RRR.##.RRR.#",
        "#..B..#.##.....#",  # Bounce block shortcut
        "#.###.####.#####",
        "#.#...####.#####",
        "#...#.####..G..#",
        "################",
    ],
    "The Spiral": [
        "################",
        "#S.............#",
        "#.############.#",
        "#.#..........#.#",
        "#.#.########.#.#",
        "#.#.#......#.#.#",
        "#.#.#.####.#.#.#",
        "#.#.#.#G.#.#.#.#",
        "#.#.#.##.#.#.#.#",
        "#.#.#....#.#.#.#",
        "#.#.######.#.#.#",
        "#.#........#.#.#",
        "#.##########.#.#",
        "#.....B......#.#",  # Bounce block shortcut
        "################",
        "################",
    ]
}

INITIAL_MAZE = PRESETS["Default Chasm"]


# ---------------------------------------------------------------------------
# 16,384-Stream GPU Physics Simulation with Bounce Block Mechanics
# ---------------------------------------------------------------------------
class TurboPhysicsEnv:
    def __init__(self, maze_ascii, num_envs=16384, device=device):
        self.num_envs = num_envs
        self.max_steps = 40
        self.device = device
        self.load_ascii_map(maze_ascii)

        # 64 Omnidirectional Actions: 16 Angles x 4 Powers
        self.num_angles = 16
        self.num_powers = 4
        self.num_actions = self.num_angles * self.num_powers

        angles = torch.linspace(-math.pi, math.pi - (2 * math.pi / self.num_angles), self.num_angles, device=device)
        powers = torch.tensor([0.7, 1.4, 2.1, 2.8], device=device)

        action_table = []
        for p in powers:
            for a in angles:
                dy = torch.sin(a) * p
                dx = torch.cos(a) * p
                action_table.append(torch.stack([dy, dx]))
        self.action_vectors = torch.stack(action_table).to(device)

        self.pos = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.prev_pos = torch.zeros((num_envs, 2), dtype=torch.float
