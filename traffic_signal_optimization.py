# %%
# ============================================================
# CELL 1: Install & Import Dependencies
# ============================================================
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
import warnings
warnings.filterwarnings('ignore')

print('✅ Libraries loaded successfully')
# %%
# ============================================================
# CELL 2: Simulation Dataset Generation
# Represents real-world traffic patterns across 24 hours
# ============================================================

np.random.seed(42)
random.seed(42)

def generate_traffic_dataset(total_steps=200):
    """
    Generate a synthetic dataset that simulates 24-hour traffic patterns.
    Each row = one time step.
    Columns: arrival_rate_lane1..4 (vehicles/step), traffic_phase, density
    """
    records = []
    for t in range(total_steps):
        # Map time step to a phase: LOW / MEDIUM / HIGH
        progress = t / total_steps
        if progress < 0.25:         # Early morning - low
            phase = 'LOW'
            base_rate = np.random.uniform(0.5, 1.5)
        elif progress < 0.45:        # Morning rush - high
            phase = 'HIGH'
            base_rate = np.random.uniform(3.5, 5.5)
        elif progress < 0.65:        # Midday - medium
            phase = 'MEDIUM'
            base_rate = np.random.uniform(1.8, 3.2)
        elif progress < 0.85:        # Evening rush - high
            phase = 'HIGH'
            base_rate = np.random.uniform(4.0, 6.0)
        else:                        # Night - low
            phase = 'LOW'
            base_rate = np.random.uniform(0.3, 1.2)

        records.append({
            'time_step': t,
            'phase': phase,
            'arrival_rate_lane1': round(base_rate * np.random.uniform(0.8, 1.2), 2),
            'arrival_rate_lane2': round(base_rate * np.random.uniform(0.8, 1.2), 2),
            'arrival_rate_lane3': round(base_rate * np.random.uniform(0.8, 1.2), 2),
            'arrival_rate_lane4': round(base_rate * np.random.uniform(0.8, 1.2), 2),
            'density': round(base_rate / 6.0, 3)   # normalized 0-1
        })

    df = pd.DataFrame(records)
    return df

dataset = generate_traffic_dataset(total_steps=200)
print(f'📊 Dataset shape: {dataset.shape}')
print(f'\nPhase distribution:')
print(dataset['phase'].value_counts())
print(f'\nFirst 5 rows:')
dataset.head()
# %%
# ============================================================
# CELL 3: Visualize Dataset Traffic Patterns
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

phase_colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
colors = dataset['phase'].map(phase_colors)

ax1 = axes[0]
ax1.bar(dataset['time_step'], dataset['arrival_rate_lane1'], color=colors, alpha=0.7, width=1.0)
ax1.set_title('Traffic Arrival Rate - Lane 1 (Color = Phase)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Vehicles / Step')
patches = [mpatches.Patch(color=c, label=p) for p, c in phase_colors.items()]
ax1.legend(handles=patches)

ax2 = axes[1]
for i, lane in enumerate(['arrival_rate_lane1','arrival_rate_lane2',
                           'arrival_rate_lane3','arrival_rate_lane4'], 1):
    ax2.plot(dataset['time_step'], dataset[lane], label=f'Lane {i}', alpha=0.75)
ax2.set_title('All Lane Arrival Rates Over Time', fontsize=13, fontweight='bold')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Vehicles / Step')
ax2.legend()

plt.tight_layout()
plt.savefig('traffic_dataset_pattern.png', dpi=120, bbox_inches='tight')
plt.show()
print('✅ Dataset pattern visualized')
# %%
# ============================================================
# CELL 4: Signal & Queue Classes
# ============================================================

class Lane:
    """Represents a single lane with its own queue."""
    def __init__(self, lane_id):
        self.lane_id     = lane_id
        self.queue       = deque()       # each item = vehicle arrival time
        self.waiting_times = []          # completed vehicle waiting times
        self.total_processed = 0
        self.total_fuel_wasted = 0.0    # litres approximation

    def arrive(self, n_vehicles, current_time):
        """n_vehicles arrive at current_time."""
        for _ in range(int(n_vehicles)):
            self.queue.append(current_time)

    def discharge(self, n_vehicles, current_time, discharge_rate=2):
        """Discharge up to n_vehicles when signal is GREEN."""
        discharged = 0
        while self.queue and discharged < discharge_rate:
            arrival_t = self.queue.popleft()
            wait = current_time - arrival_t
            self.waiting_times.append(wait)
            self.total_processed += 1
            # Fuel approximation: 0.05 L per second of idle waiting
            self.total_fuel_wasted += wait * 0.05
            discharged += 1
        return discharged

    @property
    def queue_length(self):
        return len(self.queue)

    @property
    def avg_wait(self):
        return np.mean(self.waiting_times) if self.waiting_times else 0.0


class TrafficSignal:
    """
    A single traffic signal controlling 4 lanes.
    Green times for each lane = PSO-optimized parameters.
    """
    def __init__(self, junction_id, green_times):
        self.junction_id  = junction_id
        self.green_times  = list(green_times)   # [g1, g2, g3, g4] in steps
        self.lanes        = [Lane(i) for i in range(4)]
        self.current_phase = 0                  # which lane has green
        self.phase_timer   = 0
        self.emergency_override = None          # lane index or None
        self.emergency_countdown = 0

    def trigger_emergency(self, lane_idx, duration=5):
        """Give immediate green to emergency lane."""
        self.emergency_override  = lane_idx
        self.emergency_countdown = duration

    def active_lane(self):
        """Return index of lane currently on GREEN."""
        if self.emergency_override is not None:
            return self.emergency_override
        return self.current_phase

    def step(self, arrival_rates, current_time, discharge_rate=2):
        """
        One time step:
        1. Vehicles arrive
        2. Green lane discharges
        3. Advance phase timer
        Returns vehicles that left this junction (for multi-junction flow)
        """
        # 1. Arrivals
        for i, lane in enumerate(self.lanes):
            lane.arrive(arrival_rates[i], current_time)

        # 2. Discharge green lane
        green = self.active_lane()
        exiting = self.lanes[green].discharge(discharge_rate, current_time, discharge_rate)

        # 3. Advance phase / emergency countdown
        if self.emergency_override is not None:
            self.emergency_countdown -= 1
            if self.emergency_countdown <= 0:
                self.emergency_override  = None
                self.emergency_countdown = 0
        else:
            self.phase_timer += 1
            if self.phase_timer >= self.green_times[self.current_phase]:
                self.phase_timer   = 0
                self.current_phase = (self.current_phase + 1) % 4

        return exiting

print('✅ Lane and TrafficSignal classes defined')
# %%
# ============================================================
# CELL 5: Full Simulation Engine
# ============================================================

def run_simulation(green_times_j1, green_times_j2, dataset,
                   emergency_prob=0.03, discharge_rate=2,
                   verbose=False):
    """
    Run full multi-junction simulation.

    Parameters
    ----------
    green_times_j1 : list[4]  green durations for junction 1
    green_times_j2 : list[4]  green durations for junction 2
    dataset        : DataFrame with per-step arrival rates
    emergency_prob : probability per step of an emergency vehicle
    discharge_rate : vehicles discharged per green step

    Returns
    -------
    dict with avg_waiting_time, queue_lengths, total_vehicles_processed,
         total_fuel_wasted, emergency_events, per_junction details
    """
    j1 = TrafficSignal(junction_id=1, green_times=green_times_j1)
    j2 = TrafficSignal(junction_id=2, green_times=green_times_j2)

    emergency_log = []
    phase_log     = []

    total_steps = len(dataset)

    for t in range(total_steps):
        row = dataset.iloc[t]
        phase = row['phase']
        phase_log.append(phase)

        # --- Arrival rates from dataset for Junction 1 ---
        rates_j1 = [
            row['arrival_rate_lane1'],
            row['arrival_rate_lane2'],
            row['arrival_rate_lane3'],
            row['arrival_rate_lane4'],
        ]

        # --- Emergency vehicle event ---
        if random.random() < emergency_prob:
            ev_junction = random.choice([j1, j2])
            ev_lane     = random.randint(0, 3)
            ev_junction.trigger_emergency(ev_lane, duration=5)
            emergency_log.append({
                'time': t,
                'junction': ev_junction.junction_id,
                'lane': ev_lane
            })
            if verbose:
                print(f'🚑 t={t}: Emergency @ J{ev_junction.junction_id} Lane {ev_lane+1}')

        # --- Step Junction 1 ---
        exiting_j1 = j1.step(rates_j1, t, discharge_rate)

        # --- Multi-Junction: vehicles exiting J1 arrive at J2 (lane 0) ---
        # Spread exiting vehicles across J2 lanes with noise
        flow_to_j2 = exiting_j1
        rates_j2 = [
            row['arrival_rate_lane1'] * 0.5 + flow_to_j2 * 0.4,
            row['arrival_rate_lane2'] * 0.5 + flow_to_j2 * 0.3,
            row['arrival_rate_lane3'] * 0.5 + flow_to_j2 * 0.2,
            row['arrival_rate_lane4'] * 0.5 + flow_to_j2 * 0.1,
        ]

        # --- Step Junction 2 ---
        j2.step(rates_j2, t, discharge_rate)

    # ============================================================
    # FIXED FINAL OUTPUT FORMAT
    # ============================================================
    all_wait = []
    for lane in j1.lanes + j2.lanes:
        all_wait.extend(lane.waiting_times)

    avg_waiting_time = round(np.mean(all_wait), 4) if all_wait else 0.0

    queue_lengths = {
        'junction_1': {f'lane_{i+1}': j1.lanes[i].queue_length for i in range(4)},
        'junction_2': {f'lane_{i+1}': j2.lanes[i].queue_length for i in range(4)},
    }

    total_vehicles_processed = (
        sum(l.total_processed for l in j1.lanes) +
        sum(l.total_processed for l in j2.lanes)
    )

    total_fuel_wasted = round(
        sum(l.total_fuel_wasted for l in j1.lanes) +
        sum(l.total_fuel_wasted for l in j2.lanes), 3
    )

    return {
        # ---- MANDATORY FIXED OUTPUT ----
        'avg_waiting_time':         avg_waiting_time,
        'queue_lengths':            queue_lengths,
        'total_vehicles_processed': total_vehicles_processed,
        # ---- ADDITIONAL METRICS ----
        'total_fuel_wasted_litres': total_fuel_wasted,
        'emergency_events':         len(emergency_log),
        'emergency_log':            emergency_log,
        'phase_log':                phase_log,
        'junction_1':               j1,
        'junction_2':               j2,
    }

print('✅ run_simulation() defined')
# %%
# ============================================================
# CELL 6: PSO Optimizer
# ============================================================

class PSO:
    """
    Particle Swarm Optimization to find optimal green-time allocations
    for both junctions simultaneously.

    Particle dimension = 8  (4 green times for J1 + 4 for J2)
    Objective          = minimize average waiting time
    """

    def __init__(self, dataset, n_particles=20, n_iterations=30,
                 dim=8, min_green=3, max_green=20,
                 w=0.6, c1=1.5, c2=1.5):
        self.dataset      = dataset
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.dim          = dim
        self.min_green    = min_green
        self.max_green    = max_green
        self.w            = w    # inertia
        self.c1           = c1   # cognitive
        self.c2           = c2   # social

        # Initialise particles
        self.positions  = np.random.uniform(min_green, max_green,
                                            (n_particles, dim))
        self.velocities = np.random.uniform(-2, 2, (n_particles, dim))
        self.pbest_pos  = self.positions.copy()
        self.pbest_val  = np.full(n_particles, np.inf)
        self.gbest_pos  = None
        self.gbest_val  = np.inf
        self.history    = []

    def _fitness(self, particle):
        g1 = np.clip(np.round(particle[:4]).astype(int),
                     self.min_green, self.max_green)
        g2 = np.clip(np.round(particle[4:]).astype(int),
                     self.min_green, self.max_green)
        result = run_simulation(g1, g2, self.dataset,
                                emergency_prob=0.03, verbose=False)
        return result['avg_waiting_time']

    def optimize(self):
        print(f'🔁 Starting PSO: {self.n_particles} particles × {self.n_iterations} iterations')
        print('-' * 60)

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                val = self._fitness(self.positions[i])

                if val < self.pbest_val[i]:
                    self.pbest_val[i] = val
                    self.pbest_pos[i] = self.positions[i].copy()

                if val < self.gbest_val:
                    self.gbest_val = val
                    self.gbest_pos = self.positions[i].copy()

            # Update velocities and positions
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            self.velocities = (
                self.w  * self.velocities
                + self.c1 * r1 * (self.pbest_pos - self.positions)
                + self.c2 * r2 * (self.gbest_pos - self.positions)
            )
            self.positions = np.clip(
                self.positions + self.velocities,
                self.min_green, self.max_green
            )

            self.history.append(self.gbest_val)
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f'  Iter {iteration+1:3d}/{self.n_iterations} │ '
                      f'Best Avg Wait = {self.gbest_val:.4f} steps')

        print('-' * 60)
        print(f'✅ PSO complete. Best Avg Wait = {self.gbest_val:.4f} steps')
        return self.gbest_pos, self.gbest_val

print('✅ PSO class defined')
# %%
# ============================================================
# CELL 7: Run PSO Optimization
# ============================================================

pso = PSO(
    dataset      = dataset,
    n_particles  = 20,
    n_iterations = 30,
    min_green    = 3,
    max_green    = 20
)

best_params, best_fitness = pso.optimize()

best_green_j1 = np.clip(np.round(best_params[:4]).astype(int), 3, 20)
best_green_j2 = np.clip(np.round(best_params[4:]).astype(int), 3, 20)

print(f'\n🏆 Optimal Green Times:')
print(f'  Junction 1: {best_green_j1} steps')
print(f'  Junction 2: {best_green_j2} steps')
# %%
# ============================================================
# CELL 8: Plot PSO Convergence
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(pso.history, color='royalblue', linewidth=2, label='Global Best')
plt.title('PSO Convergence Curve — Avg Waiting Time', fontsize=14, fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Average Waiting Time (steps)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pso_convergence.png', dpi=120, bbox_inches='tight')
plt.show()
print('✅ PSO convergence plotted')
# %%
# ============================================================
# CELL 9: Final Simulation with Optimized Parameters
# ============================================================

final_result = run_simulation(
    green_times_j1 = best_green_j1,
    green_times_j2 = best_green_j2,
    dataset        = dataset,
    emergency_prob = 0.03,
    discharge_rate = 2,
    verbose        = True
)

print(f'\n📋 Simulation complete.')
print(f'  Emergency events occurred: {final_result["emergency_events"]}')
# %%
# ============================================================
# CELL 10: *** FIXED FINAL OUTPUT FORMAT ***
# ============================================================

print('=' * 60)
print('           📊 FINAL SIMULATION OUTPUT')
print('=' * 60)

print(f'\n1️⃣  Average Waiting Time : {final_result["avg_waiting_time"]} steps')

print(f'\n2️⃣  Queue Lengths (end of simulation):')
for junc, lanes in final_result['queue_lengths'].items():
    print(f'    {junc.upper()}:')
    for lane, length in lanes.items():
        print(f'      {lane}: {length} vehicles')

print(f'\n3️⃣  Total Vehicles Processed : {final_result["total_vehicles_processed"]}')

print('=' * 60)

# Additional metrics
print(f'\n🔖 Additional Metrics:')
print(f'   Fuel Wasted (approx)   : {final_result["total_fuel_wasted_litres"]} litres')
print(f'   Emergency Events       : {final_result["emergency_events"]}')
print(f'   Optimal Green J1       : {best_green_j1}')
print(f'   Optimal Green J2       : {best_green_j2}')
# %%
# ============================================================
# CELL 11: Visualize Queue Lengths
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (junc_key, lanes) in zip(axes, final_result['queue_lengths'].items()):
    lane_names  = list(lanes.keys())
    lane_values = list(lanes.values())
    bars = ax.bar(lane_names, lane_values, color=['#e74c3c','#3498db','#2ecc71','#f39c12'],
                  edgecolor='black', linewidth=0.8)
    ax.set_title(f'{junc_key.replace("_"," ").title()} — Final Queue Lengths',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Vehicles in Queue')
    ax.set_ylim(0, max(lane_values) * 1.3 + 1)
    for bar, val in zip(bars, lane_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('queue_lengths.png', dpi=120, bbox_inches='tight')
plt.show()
print('✅ Queue lengths visualized')
# %%
# ============================================================
# CELL 12: Dynamic Traffic Phase Distribution
# ============================================================

phase_counts = pd.Series(final_result['phase_log']).value_counts()
colors_pie   = [phase_colors[p] for p in phase_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
axes[0].pie(phase_counts, labels=phase_counts.index, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 12})
axes[0].set_title('Traffic Phase Distribution', fontsize=13, fontweight='bold')

# Per-junction avg wait
j1_waits = [l.avg_wait for l in final_result['junction_1'].lanes]
j2_waits = [l.avg_wait for l in final_result['junction_2'].lanes]

x = np.arange(4)
w = 0.35
axes[1].bar(x - w/2, j1_waits, w, label='Junction 1', color='steelblue', alpha=0.85)
axes[1].bar(x + w/2, j2_waits, w, label='Junction 2', color='darkorange', alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Lane 1','Lane 2','Lane 3','Lane 4'])
axes[1].set_title('Average Waiting Time per Lane & Junction', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Avg Wait (steps)')
axes[1].legend()

plt.tight_layout()
plt.savefig('phase_and_wait.png', dpi=120, bbox_inches='tight')
plt.show()
print('✅ Phase & waiting time chart plotted')
# %%
# ============================================================
# CELL 13: Before vs After PSO Comparison
# ============================================================

# Baseline: equal green times of 10 steps each junction
baseline_result = run_simulation(
    green_times_j1 = [10, 10, 10, 10],
    green_times_j2 = [10, 10, 10, 10],
    dataset        = dataset,
    emergency_prob = 0.03,
    verbose        = False
)

metrics = ['Avg Waiting Time', 'Total Vehicles Processed', 'Fuel Wasted (L)']
before  = [
    baseline_result['avg_waiting_time'],
    baseline_result['total_vehicles_processed'],
    baseline_result['total_fuel_wasted_litres']
]
after   = [
    final_result['avg_waiting_time'],
    final_result['total_vehicles_processed'],
    final_result['total_fuel_wasted_litres']
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, b, a in zip(axes, metrics, before, after):
    ax.bar(['Baseline', 'PSO Optimized'], [b, a],
           color=['#e74c3c', '#2ecc71'], edgecolor='black')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel('Value')
    for i, val in enumerate([b, a]):
        ax.text(i, val * 1.02, f'{val:.2f}', ha='center', fontsize=10)

plt.suptitle('🚦 Baseline vs PSO-Optimized Performance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('baseline_vs_pso.png', dpi=120, bbox_inches='tight')
plt.show()

improvement = ((baseline_result['avg_waiting_time'] - final_result['avg_waiting_time'])
               / baseline_result['avg_waiting_time'] * 100)
print(f'\n📈 PSO Improvement in Avg Waiting Time: {improvement:.2f}%')
# %%
# ============================================================
# CELL 14: Emergency Vehicle Event Summary
# ============================================================

if final_result['emergency_log']:
    ev_df = pd.DataFrame(final_result['emergency_log'])
    print(f'🚑 Emergency Vehicle Events ({len(ev_df)} total):')
    print(ev_df.to_string(index=False))

    # Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ev_df['junction'].value_counts().plot.bar(ax=axes[0], color='tomato', edgecolor='black')
    axes[0].set_title('Emergency Events by Junction', fontweight='bold')
    axes[0].set_xlabel('Junction')
    axes[0].set_ylabel('Count')

    ev_df['lane'].value_counts().sort_index().plot.bar(ax=axes[1], color='steelblue', edgecolor='black')
    axes[1].set_title('Emergency Events by Lane', fontweight='bold')
    axes[1].set_xlabel('Lane (0-indexed)')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('emergency_events.png', dpi=120, bbox_inches='tight')
    plt.show()
else:
    print('ℹ️ No emergency events occurred in this run.')
# %%
# ============================================================
# CELL 15: Summary Table
# ============================================================

summary = pd.DataFrame({
    'Metric': [
        'Avg Waiting Time (steps)',
        'Total Vehicles Processed',
        'Fuel Wasted (litres)',
        'Emergency Events',
        'PSO Improvement (%)',
        'Optimal Green J1 (steps)',
        'Optimal Green J2 (steps)',
    ],
    'Baseline': [
        round(baseline_result['avg_waiting_time'], 4),
        baseline_result['total_vehicles_processed'],
        round(baseline_result['total_fuel_wasted_litres'], 3),
        baseline_result['emergency_events'],
        '-',
        '[10, 10, 10, 10]',
        '[10, 10, 10, 10]',
    ],
    'PSO Optimized': [
        round(final_result['avg_waiting_time'], 4),
        final_result['total_vehicles_processed'],
        round(final_result['total_fuel_wasted_litres'], 3),
        final_result['emergency_events'],
        round(improvement, 2),
        str(list(best_green_j1)),
        str(list(best_green_j2)),
    ]
})

print('\n' + '='*70)
print('                    COMPLETE RESULTS SUMMARY')
print('='*70)
print(summary.to_string(index=False))
print('='*70)
# %%
# ============================================================
# CELL 16: Open Interactive Web Dashboard
# ============================================================

from pathlib import Path
import webbrowser

dashboard_path = Path(r'C:\Users\lokes\Downloads\pso_traffic_dashboard_v5 (1).html')

if not dashboard_path.is_file():
    raise FileNotFoundError(f'Dashboard not found: {dashboard_path}')

dashboard_url = dashboard_path.as_uri()
print(f'Opening dashboard: {dashboard_url}')
webbrowser.open(dashboard_url)
