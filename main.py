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
# Device Configuration & Hardware Auto-Tuning
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


def get_auto_parallelism(dev):
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        sm_count = props.multi_processor_count
        name = props.name

        if vram_gb >= 15.0:
            num_envs = 65536
            micro_batch = 16
        elif vram_gb >= 10.0:
            num_envs = 32768
            micro_batch = 16
        elif vram_gb >= 6.0:
            num_envs = 16384
            micro_batch = 12
        else:
            num_envs = 8192
            micro_batch = 8

        print(f"[*] Detected GPU: {name} | VRAM: {vram_gb:.1f} GB | SMs: {sm_count}")
        print(f"[+] High-Precision Engine: {num_envs:,} Streams | Batch Step: {micro_batch}")
        return num_envs, micro_batch
    else:
        print("[*] Running on CPU Mode")
        return 1024, 4


AUTO_NUM_ENVS, AUTO_MICRO_BATCH = get_auto_parallelism(device)

# ---------------------------------------------------------------------------
# Built-in Maze Presets (Trick Sections & Chasms)
# ---------------------------------------------------------------------------
# '#' = Low Wall (Jumpable if cleared; landing on it = DEATH)
# 'R' = Tall Red Tower (Blocks all flight; contact = DEATH)
# 'B' = Bounce Block (Super-altitude launch: soar OVER Red Towers!)
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
        "#..B####....####",  # Bounce Block 'B' in front of Red Barricade!
        "#.RRRRRRRR#.####",  # Vaultable with Super Altitude from 'B'!
        "#..........#.###",
        "############.###",
        "############.###",
        "############..G#",  # Safe inside boundary wall
        "################",
    ],
    "Chicane Gauntlet": [
        "################",
        "#S.#...#.......#",
        "##.###.#.#####.#",
        "#....#.#.#...#.#",
        "#.##.#.###.#.#.#",
        "#..#...#...#.#.#",
        "##.#####.###.#.#",  # Tight 1-block trick chicane
        "#..#...#.#...#.#",
        "#.##.#.#.###.#.#",
        "#..B.#...#...#.#",  # Trick pad
        "####.#####.#####",
        "#....#.........#",
        "#.####.RRRRRR#.#",
        "#.#....#.....#.#",
        "#...##...###..G#",
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
        "#............#.#",
        "################",
        "################",
    ]
}

INITIAL_MAZE = PRESETS["Default Chasm"]


# ---------------------------------------------------------------------------
# High-Precision GPU Swarm Simulation (4 Discrete Power Tiers)
# ---------------------------------------------------------------------------
class PrecisionPhysicsEnv:
    def __init__(self, maze_ascii, num_envs=AUTO_NUM_ENVS, device=device):
        self.num_envs = num_envs
        self.max_steps = 45
        self.device = device
        self.load_ascii_map(maze_ascii)

        # 64 Actions: 16 Angles x 4 Deliberate Jump Powers
        # Power 0.4: Micro-hop for tight trick sections / chicane turns
        # Power 0.9: Short hop (1 block advance)
        # Power 1.7: Medium leap (vaults 1-block low wall / chasm)
        # Power 2.8: Super leap (wide chasm clearance)
        self.num_angles = 16
        self.num_powers = 4
        self.num_actions = self.num_angles * self.num_powers

        angles = torch.linspace(-math.pi, math.pi - (2 * math.pi / self.num_angles), self.num_angles, device=device)
        powers = torch.tensor([0.4, 0.9, 1.7, 2.8], device=device)

        action_table = []
        angle_list = []
        power_list = []
        for p in powers:
            for a in angles:
                dy = torch.sin(a) * p
                dx = torch.cos(a) * p
                action_table.append(torch.stack([dy, dx]))
                angle_list.append(a)
                power_list.append(p)

        self.action_vectors = torch.stack(action_table).to(device)
        self.act_angles = torch.tensor(angle_list, device=device)
        self.act_powers = torch.tensor(power_list, device=device)

        self.pos = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.last_act = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # High-definition sub-tile landing counts: 32x32 resolution
        self.map_landing_counts = torch.zeros((self.H * 2, self.W * 2), dtype=torch.float32, device=device)

        self.sample_t = torch.linspace(0.0625, 0.9375, 10, device=device).view(1, 10, 1)

    def load_ascii_map(self, maze_ascii):
        self.H = len(maze_ascii)
        self.W = len(maze_ascii[0])

        grid_np = np.zeros((self.H, self.W), dtype=np.int64)
        start_pos, goal_pos = None, None

        for r, row in enumerate(maze_ascii):
            for c, char in enumerate(row):
                if char == '#':
                    grid_np[r, c] = 1
                elif char == 'R':
                    grid_np[r, c] = 2
                elif char == 'B':
                    grid_np[r, c] = 3
                elif char == 'S':
                    start_pos = (float(r) + 0.5, float(c) + 0.5)
                elif char == 'G':
                    goal_pos = (float(r) + 0.5, float(c) + 0.5)

        self.set_full_map(grid_np, start_pos, goal_pos)

    def set_full_map(self, grid_np, start_pos, goal_pos):
        self.grid = torch.tensor(grid_np, dtype=torch.long, device=self.device)
        self.start_coord = torch.tensor(start_pos, dtype=torch.float32, device=self.device)
        self.goal_coord = torch.tensor(goal_pos, dtype=torch.float32, device=self.device)
        free_coords = torch.nonzero((self.grid == 0) | (self.grid == 3)).float() + 0.5
        self.free_cells = free_coords.to(self.device)
        self.map_landing_counts = torch.zeros((self.H * 2, self.W * 2), dtype=torch.float32, device=self.device)
        self.ideal_straight_dist = torch.norm(self.goal_coord - self.start_coord).item()

    def reset(self):
        starts = self.start_coord.unsqueeze(0).expand(self.num_envs, -1).clone()
        half = self.num_envs // 2
        if len(self.free_cells) > 0:
            rand_idx = torch.randint(0, len(self.free_cells), (half,), device=self.device)
            starts[:half] = self.free_cells[rand_idx]

        self.pos.copy_(starts)
        self.last_act.fill_(-1)
        self.step_count.zero_()

    def get_quad_bins(self, pos):
        """Converts continuous coordinates into 32x32 sub-tile spatial bins (0.5x0.5 per bin)."""
        qr = (pos[:, 0] * 2.0).long().clamp(0, self.H * 2 - 1)
        qc = (pos[:, 1] * 2.0).long().clamp(0, self.W * 2 - 1)
        return qr, qc

    def step(self, action_ids, goal_solved=False):
        cur_r = self.pos[:, 0].long().clamp(0, self.H - 1)
        cur_c = self.pos[:, 1].long().clamp(0, self.W - 1)

        # Check if launching from a Bounce Block (Super-Altitude!)
        from_bounce = (self.grid[cur_r, cur_c] == 3)

        self.step_count += 1
        delta = self.action_vectors[action_ids]
        power = self.act_powers[action_ids]
        angle = self.act_angles[action_ids]

        target_pos = self.pos + delta
        ty = target_pos[:, 0]
        tx = target_pos[:, 1]

        # 1. Boundary & Wall Landing Check
        oob = (ty < 0.5) | (ty >= self.H - 0.5) | (tx < 0.5) | (tx >= self.W - 0.5)
        safe_y = ty.long().clamp(0, self.H - 1)
        safe_x = tx.long().clamp(0, self.W - 1)
        landing_type = self.grid[safe_y, safe_x]
        landing_deadly = oob | (landing_type == 1) | (landing_type == 2)

        # 2. Mid-Air Red Block Collision Check (Cleared if launching from Bounce Pad!)
        mid_pts = self.pos.unsqueeze(1) + delta.unsqueeze(1) * self.sample_t
        my = mid_pts[..., 0].long().clamp(0, self.H - 1)
        mx = mid_pts[..., 1].long().clamp(0, self.W - 1)
        is_red_intercept = (self.grid[my, mx] == 2).any(dim=1)
        hit_red_midair = (~from_bounce) & is_red_intercept

        died = landing_deadly | hit_red_midair

        new_dist = torch.norm(target_pos - self.goal_coord, dim=-1)
        reached_goal = (~died) & (new_dist < 0.65)
        timeout = self.step_count >= self.max_steps

        # 3. Anti-U-Turn Directional Check:
        # Penalizes an immediate 180° reversal without punishing forward/lateral micro-hops!
        has_prev_act = self.last_act >= 0
        prev_angles = torch.where(has_prev_act, self.act_angles[self.last_act.clamp(min=0)], angle)
        angle_diff = torch.cos(angle - prev_angles)
        is_u_turn = has_prev_act & (angle_diff < -0.85)

        # 4. Reward Formulation (Balanced for both micro-hops and long leaps)
        tgt_qr, tgt_qc = self.get_quad_bins(target_pos)
        landings = self.map_landing_counts[tgt_qr, tgt_qc]

        if not goal_solved:
            # DISCOVERY MODE: Micro-hops and precision hops are NOT penalized!
            # Rewards reaching new/unvisited sub-tile quadrants
            subtile_bonus = 8.0 / (1.0 + 0.1 * landings)
            bounce_launch_bonus = torch.where(from_bounce, torch.tensor(3.0, device=self.device), torch.tensor(0.0, device=self.device))
            reward = subtile_bonus + bounce_launch_bonus - 0.02
        else:
            # SPEEDRUN MODE: Modest step cost to encourage pruning excess leaps
            reward = -0.30 - (0.02 * power)

        # Heavy penalties for death, timeout, and doubling back
        reward = torch.where(is_u_turn, reward - 3.5, reward)
        reward = torch.where(died, torch.full_like(reward, -10.0), reward)
        reward = torch.where(timeout & (~reached_goal), torch.full_like(reward, -12.0), reward)
        reward = torch.where(reached_goal, reward + 100.0, reward)

        dones = died | reached_goal | timeout

        self.last_act = torch.where(died, torch.full_like(self.last_act, -1), action_ids)
        self.pos = torch.where(died.unsqueeze(-1), self.pos, target_pos)

        survivors = (~died) & (~reached_goal)
        if survivors.any():
            s_idx = torch.nonzero(survivors).view(-1)
            sqr = tgt_qr[s_idx]
            sqc = tgt_qc[s_idx]
            landed_mask = torch.zeros((self.H * 2, self.W * 2), dtype=torch.bool, device=self.device)
            landed_mask[sqr, sqc] = True
            self.map_landing_counts += landed_mask.float()

        if dones.any():
            done_idx = torch.nonzero(dones).view(-1)
            n = len(done_idx)
            new_starts = self.start_coord.unsqueeze(0).expand(n, -1).clone()

            half = n // 2
            if len(self.free_cells) > 0 and half > 0:
                rand_idx = torch.randint(0, len(self.free_cells), (half,), device=self.device)
                new_starts[:half] = self.free_cells[rand_idx]

            self.pos[dones] = new_starts
            self.last_act[dones] = -1
            self.step_count[dones] = 0

        return cur_r, cur_c, safe_y, safe_x, reward, dones, reached_goal


# ---------------------------------------------------------------------------
# 262 KB GPU L2-Cache Bellman Engine (32x32 Spatial Resolution)
# ---------------------------------------------------------------------------
class BellmanTrainer:
    def __init__(self, maze_ascii):
        self.num_envs = AUTO_NUM_ENVS
        self.micro_batch = AUTO_MICRO_BATCH
        self.device = device
        self.env = PrecisionPhysicsEnv(maze_ascii, num_envs=self.num_envs, device=device)

        self.H = self.env.H
        self.W = self.env.W
        self.A = self.env.num_actions

        # High-definition Q-Tensor: Q[32, 32, 64] = 262 KB (Fits 100% in GPU L2 Cache!)
        self.Q = torch.zeros((self.H * 2, self.W * 2, self.A), dtype=torch.float32, device=device)

        self.gamma = 0.98
        self.lr = 0.25
        self.epsilon = 0.15

        self.total_steps_counter = 0
        self.total_goals_reached = 0

        # Verified global champion route
        self.best_route = None

        self.lock = threading.Lock()
        self.shared_q_cpu = self.Q.cpu().clone()

        self.running = True
        self.thread = threading.Thread(target=self._bellman_loop, daemon=True)
        self.thread.start()

    def get_q_cpu(self):
        with self.lock:
            return self.shared_q_cpu

    def get_best_route(self):
        with self.lock:
            return self.best_route

    def create_snapshot(self):
        with self.lock:
            return {
                'grid': self.env.grid.cpu().numpy().copy(),
                'start_coord': self.env.start_coord.cpu().tolist(),
                'goal_coord': self.env.goal_coord.cpu().tolist(),
                'Q': self.Q.clone(),
                'best_route': dict(self.best_route) if self.best_route else None,
                'total_goals_reached': self.total_goals_reached,
                'total_steps_counter': self.total_steps_counter,
            }

    def apply_saved_map(self, grid_np, start_pos, goal_pos):
        with self.lock:
            self.env.set_full_map(grid_np, start_pos, goal_pos)
            self.Q.zero_()
            self.shared_q_cpu = self.Q.cpu().clone()
            self.best_route = None
            self.epsilon = 0.20
            self.total_goals_reached = 0

    def restore_map_only(self, snapshot):
        with self.lock:
            self.env.set_full_map(snapshot['grid'], snapshot['start_coord'], snapshot['goal_coord'])
            self.Q.zero_()
            self.shared_q_cpu = self.Q.cpu().clone()
            self.best_route = None
            self.epsilon = 0.20
            self.total_goals_reached = 0

    def restore_map_and_agent(self, snapshot):
        with self.lock:
            self.env.set_full_map(snapshot['grid'], snapshot['start_coord'], snapshot['goal_coord'])
            self.Q.copy_(snapshot['Q'])
            self.shared_q_cpu = self.Q.cpu().clone()
            self.best_route = dict(snapshot['best_route']) if snapshot['best_route'] else None
            self.epsilon = 0.10
            self.total_goals_reached = snapshot['total_goals_reached']
            self.total_steps_counter = snapshot['total_steps_counter']

    def load_new_preset(self, maze_ascii):
        with self.lock:
            self.env.load_ascii_map(maze_ascii)
            self.H = self.env.H
            self.W = self.env.W
            self.Q = torch.zeros((self.H * 2, self.W * 2, self.A), dtype=torch.float32, device=self.device)
            self.shared_q_cpu = self.Q.cpu().clone()
            self.best_route = None

    def extract_greedy_route(self):
        """Pre-flight simulates the greedy route on CPU to verify it 100% works!"""
        with self.lock:
            Q_copy = self.shared_q_cpu

        py, px = float(self.env.start_coord[0].item()), float(self.env.start_coord[1].item())
        grid_np = self.env.grid.cpu().numpy()
        deltas = self.env.action_vectors.cpu().numpy()
        powers = self.env.act_powers.cpu().numpy()
        H, W = self.H, self.W

        path = [(py, px)]
        powers_used = []
        total_pwr = 0.0
        visited_cells = set()

        for step in range(45):
            cr = int(max(0, min(H - 1, py)))
            cc = int(max(0, min(W - 1, px)))

            if (cr, cc) in visited_cells:
                return None
            visited_cells.add((cr, cc))

            # Query 32x32 spatial quadrant
            qr = int(max(0, min(H * 2 - 1, py * 2.0)))
            qc = int(max(0, min(W * 2 - 1, px * 2.0)))

            q_row = Q_copy[qr, qc]
            if q_row.max().item() <= -5.0:
                return None

            act = q_row.argmax().item()
            dy, dx = deltas[act]
            pwr = powers[act]
            powers_used.append(pwr)
            total_pwr += pwr

            was_bounce = (grid_np[cr, cc] == 3)
            ty = py + dy
            tx = px + dx

            if ty < 0.5 or ty >= H - 0.5 or tx < 0.5 or tx >= W - 0.5:
                return None

            lr = int(max(0, min(H - 1, ty)))
            lc = int(max(0, min(W - 1, tx)))
            if grid_np[lr, lc] in (1, 2):
                return None

            if not was_bounce:
                for t in np.linspace(0.0625, 0.9375, 10):
                    my = int(max(0, min(H - 1, py + dy * t)))
                    mx = int(max(0, min(W - 1, px + dx * t)))
                    if grid_np[my, mx] == 2:
                        return None

            py, px = ty, tx
            path.append((py, px))

            gy, gx = float(self.env.goal_coord[0].item()), float(self.env.goal_coord[1].item())
            if math.hypot(gy - py, gx - px) < 0.65:
                ideal_dist = math.hypot(gy - float(self.env.start_coord[0].item()), gx - float(self.env.start_coord[1].item()))
                directness = min(100.0, (ideal_dist / max(1e-5, total_pwr)) * 100.0)
                return {
                    'path': path,
                    'powers': powers_used,
                    'length': len(path) - 1,
                    'directness': directness
                }

        return None

    def _bellman_loop(self):
        try:
            self.env.reset()
            gpu_goals = torch.zeros(1, dtype=torch.long, device=self.device)

            while self.running:
                is_solved = (self.best_route is not None)

                for _ in range(self.micro_batch):
                    # Sub-tile spatial coordinates
                    cur_qr, cur_qc = self.env.get_quad_bins(self.env.pos)

                    q_vals = self.Q[cur_qr, cur_qc]
                    exploit_actions = q_vals.argmax(dim=-1)
                    random_actions = torch.randint(0, self.A, (self.num_envs,), device=self.device)
                    is_explore = torch.rand(self.num_envs, device=self.device) < self.epsilon
                    actions = torch.where(is_explore, random_actions, exploit_actions)

                    r_from, c_from, r_to, c_to, rewards, dones, reached_goal = self.env.step(
                        actions, goal_solved=is_solved
                    )
                    gpu_goals += reached_goal.sum()

                    nxt_qr, nxt_qc = self.env.get_quad_bins(self.env.pos)
                    max_next_q = self.Q[nxt_qr, nxt_qc].max(dim=-1).values
                    target = rewards + self.gamma * max_next_q * (~dones)

                    flat_idx = (cur_qr * (self.W * 2) + cur_qc) * self.A + actions
                    q_flat = self.Q.view(-1)
                    q_flat[flat_idx] = q_flat[flat_idx] * (1.0 - self.lr) + target * self.lr

                self.total_steps_counter += self.num_envs * self.micro_batch
                self.total_goals_reached += gpu_goals.item()
                gpu_goals.zero_()

                # Sync Q-tensor to host
                with self.lock:
                    self.shared_q_cpu = self.Q.cpu().clone()

                # Check if greedy path solved a new record
                route_cand = self.extract_greedy_route()
                if route_cand is not None:
                    with self.lock:
                        if self.best_route is None or route_cand['length'] < self.best_route['length'] or (route_cand['length'] == self.best_route['length'] and route_cand['directness'] > self.best_route['directness']):
                            self.best_route = route_cand

                time.sleep(0.0001)

        except Exception:
            print("\n[!] Error in Bellman Loop:")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Interactive Cyberpunk Visualizer Studio (Best-Only Precision Display)
# ---------------------------------------------------------------------------
class PrecisionVisualizer:
    def __init__(self, root, trainer):
        self.root = root
        self.trainer = trainer
        self.root.title("Quantum Bro - Precision Power Control Studio")
        self.root.configure(bg="#080C14")
        self.root.resizable(False, False)

        self.cs = 44
        self.H = trainer.H
        self.W = trainer.W

        self.active_grid = trainer.env.grid.cpu().numpy().copy()
        self.start_coord = trainer.env.start_coord.cpu().tolist()
        self.goal_coord = trainer.env.goal_coord.cpu().tolist()

        self.draft_grid = self.active_grid.copy()
        self.draft_start = list(self.start_coord)
        self.draft_goal = list(self.goal_coord)
        self.has_unsaved_changes = False

        self.last_snapshot = trainer.create_snapshot()

        # Best-Only Playback Engine
        self.active_best_path = None
        self.active_powers = []
        self.best_length = 999
        self.best_directness = 0.0

        self.anim_idx = 0
        self.anim_sub = 0
        self.total_subs = 4

        self.current_brush = 1
        self.show_heatmap = True

        self.last_bench_time = time.time()
        self.last_bench_steps = 0

        # UI Layout
        main_frame = ttk.Frame(root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            main_frame, width=self.W * self.cs, height=self.H * self.cs,
            bg="#02040A", highlightthickness=1, highlightbackground="#1E293B"
        )
        self.canvas.grid(row=0, column=0, padx=(0, 16))
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<Button-3>", self.on_canvas_erase)
        self.canvas.bind("<B3-Motion>", self.on_canvas_erase)

        panel = ttk.Frame(main_frame)
        panel.grid(row=0, column=1, sticky="n")

        self.status_banner = tk.Label(
            panel, text="🎯 PRECISION TRICK & LEAP CONTROL",
            font=("Helvetica", 10, "bold"), fg="#10B981", bg="#064E3B", padx=8, pady=4
        )
        self.status_banner.pack(fill="x", pady=(0, 6))

        train_box = tk.LabelFrame(panel, text=" Swarm Telemetry ", bg="#0F172A", fg="#94A3B8", font=("Helvetica", 9, "bold"), padx=10, pady=5)
        train_box.pack(fill="x", pady=3)

        self.speed_var = tk.StringVar(value="Speed: 0 steps/s")
        tk.Label(train_box, textvariable=self.speed_var, font=("Courier", 11, "bold"), bg="#0F172A", fg="#38BDF8").pack(anchor="w")

        self.goals_var = tk.StringVar(value="Swarm Solves: 0")
        tk.Label(train_box, textvariable=self.goals_var, font=("Courier", 10), bg="#0F172A", fg="#10B981").pack(anchor="w")

        best_box = tk.LabelFrame(panel, text=" Optimal Route Telemetry ", bg="#0F172A", fg="#94A3B8", font=("Helvetica", 9, "bold"), padx=10, pady=5)
        best_box.pack(fill="x", pady=3)

        self.record_var = tk.StringVar(value="Best Record: Calculating...")
        tk.Label(best_box, textvariable=self.record_var, font=("Courier", 11, "bold"), bg="#0F172A", fg="#F59E0B").pack(anchor="w")

        self.direct_var = tk.StringVar(value="Directness: 0.0%")
        tk.Label(best_box, textvariable=self.direct_var, font=("Courier", 10, "bold"), bg="#0F172A", fg="#10B981").pack(anchor="w")

        self.power_var = tk.StringVar(value="Leap Power: Ready")
        tk.Label(best_box, textvariable=self.power_var, font=("Courier", 9), bg="#0F172A", fg="#E2E8F0").pack(anchor="w")

        studio_box = tk.LabelFrame(panel, text=" Map Studio & Snapshots ", bg="#0F172A", fg="#94A3B8", font=("Helvetica", 9, "bold"), padx=10, pady=5)
        studio_box.pack(fill="x", pady=3)

        self.btn_save = tk.Button(
            studio_box, text="💾 Save & Apply Map",
            font=("Helvetica", 10, "bold"), bg="#059669", fg="#FFFFFF", padx=6, pady=3,
            command=self.save_and_apply_map
        )
        self.btn_save.pack(fill="x", pady=(1, 3))

        res_row = tk.Frame(studio_box, bg="#0F172A")
        res_row.pack(fill="x", pady=2)

        self.btn_res_map = tk.Button(
            res_row, text="↩️ Last Map", font=("Helvetica", 8, "bold"), bg="#1E293B", fg="#38BDF8",
            command=self.resume_last_map
        )
        self.btn_res_map.pack(side="left", padx=2, expand=True, fill="x")

        self.btn_res_all = tk.Button(
            res_row, text="⏮️ Last Map+Brain", font=("Helvetica", 8, "bold"), bg="#1E293B", fg="#A855F7",
            command=self.resume_last_map_and_agent
        )
        self.btn_res_all.pack(side="left", padx=2, expand=True, fill="x")

        b_row = tk.Frame(studio_box, bg="#0F172A")
        b_row.pack(fill="x", pady=2)
        brushes = [
            ("Wall (#)", 1, "#64748B"),
            ("Red (R)", 2, "#EF4444"),
            ("Bounce (B)", 3, "#10B981"),
            ("Floor (.)", 0, "#1E293B")
        ]
        for name, val, col in brushes:
            tk.Button(b_row, text=name, font=("Helvetica", 8, "bold"), bg=col, fg="#FFFFFF",
                      command=lambda v=val: self.set_brush(v)).pack(side="left", padx=1, expand=True, fill="x")

        p_row = tk.Frame(studio_box, bg="#0F172A")
        p_row.pack(fill="x", pady=2)
        tk.Button(p_row, text="Set Start (S)", font=("Helvetica", 8), bg="#0284C7", fg="#FFFFFF",
                  command=lambda: self.set_brush(4)).pack(side="left", padx=2, expand=True, fill="x")
        tk.Button(p_row, text="Set Goal (G)", font=("Helvetica", 8), bg="#D97706", fg="#FFFFFF",
                  command=lambda: self.set_brush(5)).pack(side="left", padx=2, expand=True, fill="x")

        ctrl_box = tk.LabelFrame(panel, text=" Studio Controls ", bg="#0F172A", fg="#94A3B8", font=("Helvetica", 9, "bold"), padx=10, pady=5)
        ctrl_box.pack(fill="x", pady=3)

        self.btn_heatmap = tk.Button(ctrl_box, text="🧠 Hide Heatmap", font=("Helvetica", 8, "bold"), bg="#4338CA", fg="#FFFFFF", command=self.toggle_heatmap)
        self.btn_heatmap.pack(fill="x", pady=2)

        preset_row = tk.Frame(ctrl_box, bg="#0F172A")
        preset_row.pack(fill="x", pady=3)
        tk.Label(preset_row, text="Preset:", bg="#0F172A", fg="#94A3B8", font=("Helvetica", 8)).pack(side="left")
        self.preset_combo = ttk.Combobox(preset_row, values=list(PRESETS.keys()), state="readonly", width=12)
        self.preset_combo.current(0)
        self.preset_combo.pack(side="left", padx=3)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_select)

        tk.Label(ctrl_box, text="Playback Delay (ms):", font=("Helvetica", 8), bg="#0F172A", fg="#94A3B8").pack(anchor="w", pady=(3, 0))
        self.speed_slider = ttk.Scale(ctrl_box, from_=10, to=150, value=35)
        self.speed_slider.pack(fill="x", pady=1)

        self.reset_playback()
        self.loop_tick()

    def set_brush(self, brush_val):
        self.current_brush = brush_val

    def mark_unsaved(self):
        self.has_unsaved_changes = True
        self.btn_save.config(text="⚠️ Save & Apply Map*", bg="#D97706")

    def save_and_apply_map(self):
        self.last_snapshot = self.trainer.create_snapshot()
        self.active_grid = self.draft_grid.copy()
        self.start_coord = list(self.draft_start)
        self.goal_coord = list(self.draft_goal)

        self.trainer.apply_saved_map(self.active_grid, self.start_coord, self.goal_coord)
        self.has_unsaved_changes = False
        self.btn_save.config(text="💾 Save Map (Synced)", bg="#059669")
        self.reset_playback()

    def resume_last_map(self):
        if self.last_snapshot is None:
            return
        snap = self.last_snapshot
        self.active_grid = snap['grid'].copy()
        self.start_coord = list(snap['start_coord'])
        self.goal_coord = list(snap['goal_coord'])

        self.draft_grid = self.active_grid.copy()
        self.draft_start = list(self.start_coord)
        self.draft_goal = list(self.goal_coord)

        self.trainer.restore_map_only(snap)
        self.has_unsaved_changes = False
        self.btn_save.config(text="💾 Save Map (Synced)", bg="#059669")
        self.reset_playback()

    def resume_last_map_and_agent(self):
        if self.last_snapshot is None:
            return
        snap = self.last_snapshot
        self.active_grid = snap['grid'].copy()
        self.start_coord = list(snap['start_coord'])
        self.goal_coord = list(snap['goal_coord'])

        self.draft_grid = self.active_grid.copy()
        self.draft_start = list(self.start_coord)
        self.draft_goal = list(self.goal_coord)

        self.trainer.restore_map_and_agent(snap)
        self.has_unsaved_changes = False
        self.btn_save.config(text="💾 Save Map (Synced)", bg="#059669")
        self.reset_playback()

    def toggle_heatmap(self):
        self.show_heatmap = not self.show_heatmap
        self.btn_heatmap.config(
            text="🧠 Hide Heatmap" if self.show_heatmap else "🧠 Show Heatmap",
            bg="#3730A3" if self.show_heatmap else "#4338CA"
        )
        self.render()

    def on_preset_select(self, event):
        name = self.preset_combo.get()
        ascii_map = PRESETS[name]

        grid_np = np.zeros((self.H, self.W), dtype=np.int64)
        for r, row in enumerate(ascii_map):
            for c, char in enumerate(row):
                if char == '#':
                    grid_np[r, c] = 1
                elif char == 'R':
                    grid_np[r, c] = 2
                elif char == 'B':
                    grid_np[r, c] = 3
                elif char == 'S':
                    self.draft_start = [r + 0.5, c + 0.5]
                elif char == 'G':
                    self.draft_goal = [r + 0.5, c + 0.5]

        self.draft_grid = grid_np
        self.mark_unsaved()
        self.render()

    def on_canvas_press(self, event):
        self.apply_brush(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.current_brush in (0, 1, 2, 3):
            self.apply_brush(event.x, event.y)

    def on_canvas_erase(self, event):
        old_b = self.current_brush
        self.current_brush = 0
        self.apply_brush(event.x, event.y)
        self.current_brush = old_b

    def apply_brush(self, ex, ey):
        c = ex // self.cs
        r = ey // self.cs
        if 0 <= r < self.H and 0 <= c < self.W:
            if r <= 0 or r >= self.H - 1 or c <= 0 or c >= self.W - 1:
                return

            if self.current_brush == 4:
                self.draft_start = [r + 0.5, c + 0.5]
                self.mark_unsaved()
                self.render()
                return
            elif self.current_brush == 5:
                self.draft_goal = [r + 0.5, c + 0.5]
                self.mark_unsaved()
                self.render()
                return

            if self.draft_grid[r, c] != self.current_brush:
                self.draft_grid[r, c] = self.current_brush
                self.mark_unsaved()
                self.render()

    def reset_playback(self):
        self.active_best_path = None
        self.active_powers = []
        self.anim_idx = 0
        self.anim_sub = 0
        self.render()

    def loop_tick(self):
        now = time.time()
        elapsed = now - self.last_bench_time
        if elapsed >= 0.5:
            current_steps = self.trainer.total_steps_counter
            speed_s = int((current_steps - self.last_bench_steps) / elapsed)
            self.speed_var.set(f"Speed: {speed_s:,} steps/s")
            self.last_bench_time = now
            self.last_bench_steps = current_steps
            self.goals_var.set(f"Swarm Solves: {self.trainer.total_goals_reached:,}")

        best = self.trainer.get_best_route()
        if best is not None:
            if self.active_best_path is None or best['length'] < self.best_length or (best['length'] == self.best_length and best['directness'] > self.best_directness):
                self.active_best_path = list(best['path'])
                self.active_powers = list(best['powers'])
                self.best_length = best['length']
                self.best_directness = best['directness']
                self.record_var.set(f"Best Record: {self.best_length} Jumps! ⭐")
                self.direct_var.set(f"Directness: {self.best_directness:.1f}%")
                self.anim_idx = 0
                self.anim_sub = 0

        # Advance Best Run Animation
        if self.active_best_path is not None and len(self.active_best_path) >= 2:
            pwr = self.active_powers[self.anim_idx] if self.anim_idx < len(self.active_powers) else 1.0
            pwr_tag = "MICRO 0.4" if pwr < 0.6 else ("SHORT 0.9" if pwr < 1.2 else ("MED 1.7" if pwr < 2.0 else "SUPER 2.8"))
            self.power_var.set(f"Step {self.anim_idx + 1}/{self.best_length}: [{pwr_tag}]")

            self.anim_sub += 1
            if self.anim_sub >= self.total_subs:
                self.anim_sub = 0
                self.anim_idx += 1
                if self.anim_idx >= len(self.active_best_path) - 1:
                    self.anim_idx = 0

        self.render()
        delay = int(self.speed_slider.get())
        self.root.after(delay, self.loop_tick)

    def render(self):
        self.canvas.delete("all")
        cs = self.cs

        val_map = None
        if self.show_heatmap:
            q_cpu = self.trainer.get_q_cpu()
            val_map = np.zeros((self.H, self.W))
            v_all = q_cpu.max(dim=-1).values.numpy()
            v_min, v_max = v_all.min(), v_all.max()
            norm_v = (v_all - v_min) / (v_max - v_min + 1e-6)
            for r in range(self.H):
                for c in range(self.W):
                    if self.active_grid[r, c] in (0, 3):
                        val_map[r, c] = (norm_v[r * 2, c * 2] + norm_v[r * 2 + 1, c * 2 + 1]) * 0.5

        # 1. Render Map Blocks & Heatmap
        for r in range(self.H):
            for c in range(self.W):
                x1, y1 = c * cs, r * cs
                x2, y2 = x1 + cs, y1 + cs
                cell_type = self.draft_grid[r, c]

                if cell_type == 2:  # RED TOWER
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#DC2626", outline="#7F1D1D", width=2)
                    self.canvas.create_line(x1 + 3, y1 + 3, x2 - 3, y2 - 3, fill="#991B1B", width=2)
                    self.canvas.create_line(x1 + 3, y2 - 3, x2 - 3, y1 + 3, fill="#991B1B", width=2)
                elif cell_type == 1:  # Low jumpable wall
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1E293B", outline="#334155")
                    self.canvas.create_text(x1 + cs // 2, y1 + cs // 2, text="#", fill="#475569", font=("Courier", 10))
                elif cell_type == 3:  # BOUNCE PAD
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#047857", outline="#10B981", width=2)
                    self.canvas.create_text(x1 + cs // 2, y1 + cs // 2, text="SPRING\n  ▲", fill="#A7F3D0", font=("Helvetica", 7, "bold"))
                else:  # Floor Corridor
                    if self.show_heatmap and val_map is not None:
                        score = val_map[r, c]
                        red = int(min(255, max(15, score * 255)))
                        green = int(min(255, max(20, (1.0 - abs(score - 0.5) * 2) * 200 + score * 55)))
                        blue = int(min(255, max(30, (1.0 - score) * 180)))
                        hex_col = f"#{red:02x}{green:02x}{blue:02x}"
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=hex_col, outline="#0F172A")
                    else:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#0F172A", outline="#1E293B")

        # 2. Render Portals
        sy, sx = self.draft_start
        gy, gx = self.draft_goal

        self.canvas.create_oval(
            (gx - 0.45) * cs, (gy - 0.45) * cs, (gx + 0.45) * cs, (gy + 0.45) * cs,
            fill="#10B981", outline="#FDE68A", width=3
        )
        self.canvas.create_text(gx * cs, gy * cs, text="★", font=("Helvetica", 14, "bold"), fill="#451A03")

        self.canvas.create_oval(
            (sx - 0.35) * cs, (sy - 0.35) * cs, (sx + 0.35) * cs, (sy + 0.35) * cs,
            fill="#06B6D4", outline="#22D3EE", width=2
        )

        # 3. ONLY RENDER THE BEST ROUTE (Straight Gold Vectors)
        if self.active_best_path is not None and len(self.active_best_path) >= 2:
            for i in range(len(self.active_best_path) - 1):
                p1 = self.active_best_path[i]
                p2 = self.active_best_path[i + 1]
                pwr = self.active_powers[i] if i < len(self.active_powers) else 1.0

                px1, py1 = p1[1] * cs, p1[0] * cs
                px2, py2 = p2[1] * cs, p2[0] * cs

                # Color-coded vector based on leap power tier
                if pwr < 0.6:
                    col = "#38BDF8"  # Micro-hop (Sky Blue)
                    w = 2
                elif pwr < 1.2:
                    col = "#10B981"  # Short hop (Emerald)
                    w = 2
                elif pwr < 2.0:
                    col = "#F59E0B"  # Medium leap (Gold)
                    w = 3
                else:
                    col = "#EC4899"  # Super leap (Hot Pink)
                    w = 3

                self.canvas.create_line(px1, py1, px2, py2, fill=col, width=w, arrow=tk.LAST)

            # 4. Draw Bro Avatar
            idx = self.anim_idx
            p1 = self.active_best_path[idx]
            p2 = self.active_best_path[idx + 1]
            t = self.anim_sub / float(self.total_subs)

            cur_y = p1[0] + (p2[0] - p1[0]) * t
            cur_x = p1[1] + (p2[1] - p1[1]) * t
            bx = cur_x * cs
            by = cur_y * cs

            rad = 11
            self.canvas.create_oval(bx - rad, by - rad, bx + rad, by + rad, fill="#F59E0B", outline="#FDE68A", width=2)
            self.canvas.create_oval(bx - 3, by - 3, bx + 3, by + 3, fill="#FFFFFF")
        else:
            self.canvas.create_text(
                (self.W * cs) // 2, (self.H * cs) // 2,
                text="🎯 GPU SWARM SOLVING TRICK SECTIONS & CHASMS...\n(~20M Steps/Sec)",
                fill="#38BDF8", font=("Helvetica", 11, "bold"), justify="center"
            )


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        trainer = BellmanTrainer(INITIAL_MAZE)

        print("[*] Launching Quantum Bro Precision Studio...")
        root = tk.Tk()
        app = PrecisionVisualizer(root, trainer)
        root.mainloop()
    except Exception:
        print("\n[!] Top-Level Crash in Main:")
        traceback.print_exc()
        input("Press Enter to exit...")
