import os
from time import sleep
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_all, Simulation, build_model_from_config, ROBOT_SCENE_PATH_KEY
from sbmpc.geometry import skew, quat_product, quat2rotm, quat_inverse
from sbmpc.solvers import RolloutGenerator
from sbmpc.sampler import MPPISampler
from sbmpc.gains import MPPIGain
from typing import Optional, Dict, Callable, List, Tuple
import time
import traceback
import logging
import numpy as np

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
jax.config.update("jax_default_matmul_precision", "high")

# Constants
SCENE_PATH = "examples/bitcraze_crazyflie_2/scene.xml"
INPUT_MAX = jnp.array([1, 2.5, 2.5, 2])
INPUT_MIN = jnp.array([0, -2.5, -2.5, -2])
GRAVITY = 9.81
INERTIA = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
INERTIA_MAT = jnp.diag(INERTIA)
SPATIAL_INTERTIA_MAT = jnp.diag(jnp.concatenate([0.027*jnp.ones(3, dtype=jnp.float32), INERTIA]))
SPATIAL_INTERTIA_MAT_INV = jnp.linalg.inv(SPATIAL_INTERTIA_MAT)
PENDULUM_LENGTH = 0.25

# Define waypoints for the mission
# (position, name, iterations, mass_factor, target_type)
# target_type: 'drone' = drone position reaches target, 'bucket' = bucket position reaches target
WAYPOINTS = [
    (jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32), "Above Water", 300, 1.0, 'drone'),  # Drone at this position
    (jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32), "Water Target", 300, 1.0, 'bucket'),  # Bucket touches water
    (jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32), "Above Water (Return)", 300, 12.1, 'drone'),  # Drone back up
    (jnp.array([0.8, 0.3, 0.3], dtype=jnp.float32), "Lift Target", 400, 12.1, 'drone'),  # Bucket at delivery
]

def compute_bucket_position(drone_pos, alpha, beta):
    """Compute bucket position from drone position and swing angles alpha (x-z) and beta (y-z)"""
    bucket_x = drone_pos[0] + PENDULUM_LENGTH * jnp.sin(alpha) * jnp.cos(beta)
    bucket_y = drone_pos[1] + PENDULUM_LENGTH * jnp.sin(beta)
    bucket_z = drone_pos[2] - PENDULUM_LENGTH * jnp.cos(alpha) * jnp.cos(beta)
    return jnp.array([bucket_x, bucket_y, bucket_z])

@jax.jit
def quadrotor_dynamics(state: jnp.array, inputs: jnp.array, params: jnp.array) -> jnp.array:
    """Quadrotor dynamics with 3D swing load (spherical pendulum) model"""
    # Extract state variables
    pos = state[0:3]
    quat = state[3:7]
    alpha = state[7]  # swing angle in x-z
    beta = state[8]   # swing angle in y-z
    vel_lin = state[9:12]
    ang_vel = state[12:15]
    alpha_dot = state[15]
    beta_dot = state[16]

    # Physical parameters
    uav_mass = 0.027  # kg
    pendulum_mass = 0.005  # kg
    cable_length = PENDULUM_LENGTH  # 0.25 m
    damping_coeff = 0.01  # damping coefficient
    total_mass = uav_mass + pendulum_mass

    # Rotation matrix and body forces
    R = quat2rotm(quat)
    f_body = jnp.array([0., 0., inputs[0]])  # Thrust in body frame

    # Trigonometric terms
    sin_a = jnp.sin(alpha)
    cos_a = jnp.cos(alpha)
    sin_b = jnp.sin(beta)
    cos_b = jnp.cos(beta)

    # Gravitational force
    f_grav = -total_mass * GRAVITY * jnp.array([0, 0, 1])

    # UAV torques
    total_torque = 1e-3 * inputs[1:4] - skew(ang_vel) @ INERTIA_MAT @ ang_vel

    # Initial estimate of UAV acceleration (without swing load reaction)
    total_force_initial = R @ f_body + f_grav
    uav_acc_initial = SPATIAL_INTERTIA_MAT_INV @ jnp.concatenate([total_force_initial, total_torque])
    uav_lin_acc_initial = uav_acc_initial[:3]

    # UAV acceleration in world frame
    ax, ay, az = uav_lin_acc_initial

    # Spherical pendulum equations (coupled alpha and beta)
    alpha_ddot = (
        - (GRAVITY / cable_length) * sin_a * cos_b
        - (damping_coeff / (pendulum_mass * cable_length**2)) * alpha_dot
        + (sin_b * beta_dot**2 - 2 * sin_a * cos_a * beta_dot**2) * cos_b
        - (cos_a / cable_length) * ax
        - (sin_a * sin_b / cable_length) * ay
        - (sin_a * cos_b / cable_length) * az
    )
    beta_ddot = (
        - (GRAVITY / cable_length) * sin_b
        - (damping_coeff / (pendulum_mass * cable_length**2)) * beta_dot
        + (sin_a * cos_a * alpha_dot**2) * sin_b
        - (cos_b / cable_length) * ay
        - (sin_b / cable_length) * az
    )

    # Compute line-of-sight vector and tension force T
    e = jnp.array([
        sin_a * cos_b,
        sin_b,
        -cos_a * cos_b
    ])
    r = cable_length * e

    # Acceleration of the payload in world frame
    a_tan = jnp.array([
        alpha_ddot * cable_length * cos_a * cos_b - alpha_dot**2 * cable_length * sin_a * cos_b - beta_dot**2 * cable_length * sin_a * cos_b,
        beta_ddot  * cable_length * cos_b - beta_dot**2  * cable_length * sin_b,
        0.
    ])
    a_load = a_tan + jnp.array([0., 0., GRAVITY])

    T = pendulum_mass * a_load 

    # Final UAV dynamics with swing load reaction
    total_force_final = (R @ f_body - uav_mass * GRAVITY * jnp.array([0,0,1]) - T)
    pend_torque = jnp.cross(r, -T)
    total_torque = total_torque + pend_torque
    uav_acc_final = SPATIAL_INTERTIA_MAT_INV @ jnp.concatenate([total_force_final, total_torque])
    uav_lin_acc_final = uav_acc_final[:3]
    uav_ang_acc_final = uav_acc_final[3:6]

    # State derivatives
    pos_dot = vel_lin
    quat_dot = 0.5 * quat_product(quat, jnp.concatenate([jnp.array([0.]), ang_vel]))
    vel_lin_dot = uav_lin_acc_final
    ang_vel_dot = uav_ang_acc_final

    # Combine all state derivatives
    state_dot = jnp.concatenate([
        pos_dot,
        quat_dot,
        jnp.array([alpha_dot, beta_dot]),  # alpha_dot, beta_dot
        vel_lin_dot,
        ang_vel_dot,
        jnp.array([alpha_ddot, beta_ddot])
    ])
    return state_dot

class LoadAwareObjective(BaseObjective):
    """Load-aware controller objective with speed control and dual targeting modes"""
    
    def __init__(self, max_velocity=0.5, velocity_penalty_weight=25.0):
        """
        Initialize objective with speed control parameters
        
        Args:
            max_velocity: Soft maximum velocity threshold (m/s)
            velocity_penalty_weight: Weight for velocity penalty
        """
        self.max_velocity = max_velocity
        self.velocity_penalty_weight = velocity_penalty_weight
    
    def compute_state_error(self, state: jnp.ndarray, reference: jnp.ndarray):
        """
        Compute state errors with support for both drone and bucket positioning
        
        Args:
            state: Current state vector
            reference: Reference vector [x, y, z, thrust, tx, ty, tz, target_type]
                     where target_type: 0.0 = bucket target, 1.0 = drone target
        """
        drone_pos = state[0:3]
        quat = state[3:7]
        alpha = state[7]
        beta = state[8]
        vel_lin = state[9:12]
        ang_vel = state[12:15]
        alpha_dot = state[15]
        beta_dot = state[16]
        
        target_pos = reference[:3]
        
        # Check if this is a drone target or bucket target
        # Using reference[7] as target type indicator (0=bucket, 1=drone)
        is_drone_target = reference[7] if len(reference) > 7 else 0.0
        
        # Calculate position error based on target type
        bucket_pos = compute_bucket_position(drone_pos, alpha, beta)
        
        # Use weighted combination: when is_drone_target=1, use drone_pos; when 0, use bucket_pos
        effective_pos = is_drone_target * drone_pos + (1.0 - is_drone_target) * bucket_pos
        pos_err = effective_pos - target_pos
        
        att_err = quat_product(quat_inverse(quat), jnp.array([1., 0., 0., 0.]))[1:4]
        
        # Velocity penalty with soft threshold
        vel_magnitude = jnp.linalg.norm(vel_lin)
        vel_excess = jnp.maximum(0, vel_magnitude - self.max_velocity)
        vel_err = vel_lin + vel_excess * vel_lin / (vel_magnitude + 1e-6)
        
        ang_vel_err = ang_vel
        
        # Only penalize large swings (using soft thresholds)
        swing_threshold = 15 * jnp.pi/180  # 15 degrees
        alpha_excess = jnp.maximum(0, jnp.abs(alpha) - swing_threshold)
        beta_excess = jnp.maximum(0, jnp.abs(beta) - swing_threshold)
        hinge_excess = jnp.array([alpha_excess, beta_excess])
        
        # Only penalize high swing rates
        rate_threshold = 0.5  # rad/s
        alpha_dot_excess = jnp.maximum(0, jnp.abs(alpha_dot) - rate_threshold)
        beta_dot_excess = jnp.maximum(0, jnp.abs(beta_dot) - rate_threshold)
        swing_rate_excess = jnp.array([alpha_dot_excess, beta_dot_excess])
        
        return pos_err, att_err, vel_err, ang_vel_err, hinge_excess, swing_rate_excess

    def running_cost(self, state: jnp.ndarray, inputs: jnp.ndarray, reference):
        pos_err, att_err, vel_err, ang_vel_err, hinge_excess, swing_rate_excess = self.compute_state_error(state, reference)
        
        return (
            35 * pos_err.transpose() @ pos_err +
            5 * att_err.transpose() @ att_err +
            self.velocity_penalty_weight * vel_err.transpose() @ vel_err +
            6 * ang_vel_err.transpose() @ ang_vel_err +
            20 * hinge_excess.transpose() @ hinge_excess +
            15 * swing_rate_excess.transpose() @ swing_rate_excess +
            (inputs - reference[3:7]).transpose() @ jnp.diag(jnp.array([8, 8, 8, 60])) @ (inputs - reference[3:7])
        )

    def final_cost(self, state, reference):
        pos_err, att_err, vel_err, ang_vel_err, hinge_excess, swing_rate_excess = self.compute_state_error(state, reference)
        
        return (
            150 * pos_err.transpose() @ pos_err +
            10 * att_err.transpose() @ att_err +
            self.velocity_penalty_weight * 2 * vel_err.transpose() @ vel_err +
            15 * ang_vel_err.transpose() @ ang_vel_err +
            30 * hinge_excess.transpose() @ hinge_excess +
            25 * swing_rate_excess.transpose() @ swing_rate_excess
        )

def create_robot_config(q_init: jnp.ndarray) -> settings.RobotConfig:
    """Create robot configuration"""
    robot_config = settings.RobotConfig()
    robot_config.robot_scene_path = SCENE_PATH
    robot_config.nq = 9  # [x, y, z, q0, q1, q2, q3, alpha, beta]
    robot_config.nv = 8  # [vx, vy, vz, wx, wy, wz, alpha_dot, beta_dot]
    robot_config.nu = 4
    robot_config.input_min = INPUT_MIN
    robot_config.input_max = INPUT_MAX
    robot_config.q_init = q_init
    return robot_config

def create_mpc_config(robot_config: settings.RobotConfig, mass: float) -> settings.Config:
    """Create MPC configuration for load-aware swinging load control"""
    config = settings.Config(robot_config)
    config.general.visualize = True
    config.MPC.dt = 0.015
    config.MPC.horizon = 60
    config.MPC.std_dev_mppi = 0.18 * jnp.array([0.06, 0.06, 0.06, 0.025])
    config.MPC.num_parallel_computations = 200
    config.MPC.initial_guess = jnp.array([mass*GRAVITY, 0., 0., 0.], dtype=jnp.float32)
    config.MPC.lambda_mpc = 120.0
    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 8
    config.MPC.gains = False
    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM
    return config

class WaypointMissionSimulation(Simulation):
    """Waypoint-based mission simulation for precise trajectory control"""
    
    def __init__(self, initial_state, model, controller, sampler, gains, 
                 waypoints: List[Tuple[jnp.array, str, int, float, str]],
                 config: settings.Config, visualize_params: Optional[Dict] = None, 
                 obstacles: bool = False):
        
        # Calculate total iterations
        total_iterations = sum(wp[2] for wp in waypoints)
        original_sim_iterations = config.sim_iterations
        config.sim_iterations = total_iterations
        
        # Initialize with first waypoint as reference
        uav_mass = 0.027  # kg
        target_type_flag = 1.0 if waypoints[0][4] == 'drone' else 0.0
        first_reference = jnp.concatenate([
            waypoints[0][0], 
            jnp.array([uav_mass * waypoints[0][3] * GRAVITY, 0., 0., 0., target_type_flag], dtype=jnp.float32)
        ])
        
        # Initialize base simulation
        super().__init__(initial_state, model, controller, sampler, gains, 
                        first_reference, config, visualize_params, obstacles)
        
        # Store waypoint information
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.waypoint_start_iters = [0]
        cumulative_iter = 0
        for wp in waypoints[:-1]:
            cumulative_iter += wp[2]
            self.waypoint_start_iters.append(cumulative_iter)
        
        self.total_iterations = total_iterations
        self.uav_mass = uav_mass
        
        # Tracking data
        self.waypoint_errors = []
        self.waypoint_completion_times = []
        
        # Restore original configuration
        config.sim_iterations = original_sim_iterations

    def get_current_waypoint_info(self):
        """Get current waypoint information"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return self.waypoints[-1]

    def get_current_reference(self):
        """Get reference for current waypoint"""
        waypoint_pos, _, _, mass_factor, target_type = self.get_current_waypoint_info()
        target_type_flag = 1.0 if target_type == 'drone' else 0.0
        return jnp.concatenate([
            waypoint_pos, 
            jnp.array([self.uav_mass * mass_factor * GRAVITY, 0., 0., 0., target_type_flag], dtype=jnp.float32)
        ])

    def check_waypoint_transition(self):
        """Check if it's time to transition to next waypoint"""
        if self.current_waypoint_idx < len(self.waypoints) - 1:
            next_transition_iter = self.waypoint_start_iters[self.current_waypoint_idx + 1]
            if self.iter >= next_transition_iter:
                # Calculate and log waypoint achievement
                current_state = self.current_state
                drone_pos = current_state[0:3]
                bucket_pos = compute_bucket_position(
                    drone_pos, current_state[7], current_state[8]
                )
                waypoint_pos, waypoint_name, _, _, target_type = self.get_current_waypoint_info()
                
                # Calculate error based on target type
                if target_type == 'drone':
                    error = float(jnp.linalg.norm(drone_pos - waypoint_pos))
                    print(f"\n✅ Waypoint '{waypoint_name}' reached (Drone position)")
                else:
                    error = float(jnp.linalg.norm(bucket_pos - waypoint_pos))
                    print(f"\n✅ Waypoint '{waypoint_name}' reached (Bucket position)")
                
                self.waypoint_errors.append(error)
                self.waypoint_completion_times.append(self.iter * self.rollout_gen.dt)
                
                print(f"   Error: {error:.4f}m, Time: {self.iter * self.rollout_gen.dt:.2f}s")
                print(f"   Drone at: ({drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f})")
                print(f"   Bucket at: ({bucket_pos[0]:.3f}, {bucket_pos[1]:.3f}, {bucket_pos[2]:.3f})")
                
                # Transition to next waypoint
                self.current_waypoint_idx += 1
                next_wp = self.waypoints[self.current_waypoint_idx]
                target_type_str = f"({next_wp[4]} target)"
                print(f"\n🎯 Navigating to waypoint '{next_wp[1]}' {target_type_str}")
                print(f"   Target: {next_wp[0]}")

    def update(self):
        """Update simulation state and compute control input"""
        # Check for waypoint transition
        self.check_waypoint_transition()
        
        # Get current reference
        current_reference = self.get_current_reference()
        
        # Compute optimal input sequence
        waypoint_name = self.get_current_waypoint_info()[1]
        time_start = time.time_ns()
        print(f"iteration {self.iter} (Waypoint: {waypoint_name})")
        
        input_sequence = self.controller.command(self.current_state, current_reference, num_steps=1).block_until_ready()
        ctrl = input_sequence[0, :].block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))

        self.input_traj[self.iter, :] = ctrl

        # Update dynamics
        self.current_state = self.model.integrate_sim(self.current_state, ctrl, self.rollout_gen.dt)
        self.state_traj[self.iter + 1, :] = self.current_state

    def simulate(self):
        """Simulate the waypoint mission"""
        if self.visualizer is not None:
            try:
                print("\n🚁 Starting Waypoint Mission")
                print(f"   Total waypoints: {len(self.waypoints)}")
                print(f"   Starting at waypoint: '{self.waypoints[0][1]}' ({self.waypoints[0][4]} target)")
                
                while self.visualizer.is_running() and self.iter < self.total_iterations:
                    if not self.paused:
                        step_start = time.time()
                        self.step()
                        self.visualizer.set_qpos(self.current_state[:self.model.get_nq()])
                        
                        if self.obstacles:
                            self.visualizer.move_obstacles(self.iter)

                        time_until_next_step = self.rollout_gen.dt - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                
                # Final waypoint error
                final_state = self.current_state
                drone_pos = final_state[0:3]
                final_bucket_pos = compute_bucket_position(
                    drone_pos, final_state[7], final_state[8]
                )
                final_waypoint = self.waypoints[-1]
                
                # Calculate final error based on target type
                if final_waypoint[4] == 'drone':
                    final_error = float(jnp.linalg.norm(drone_pos - final_waypoint[0]))
                else:
                    final_error = float(jnp.linalg.norm(final_bucket_pos - final_waypoint[0]))
                
                self.waypoint_errors.append(final_error)
                self.waypoint_completion_times.append(self.iter * self.rollout_gen.dt)
                
                print(f"\n✅ Final waypoint '{final_waypoint[1]}' reached ({final_waypoint[4]} target)")
                print(f"   Error: {final_error:.4f}m")
                
                print("\n✅ Waypoint mission completed!")
                self.visualizer.close()
                
            except Exception as err:
                tb_str = traceback.format_exc()
                logging.error("caught exception below, closing visualizer")
                logging.error(tb_str)
                self.visualizer.close()
                raise
        else:
            print("\n🚁 Starting Waypoint Mission (No Visualization)")
            while self.iter < self.total_iterations:
                self.step()
            print("\n✅ Waypoint mission completed!")

    def get_waypoint_trajectories(self):
        """Get trajectory segments for each waypoint"""
        segments = []
        for i in range(len(self.waypoints)):
            start_iter = self.waypoint_start_iters[i]
            if i < len(self.waypoints) - 1:
                end_iter = self.waypoint_start_iters[i + 1]
            else:
                end_iter = self.total_iterations
            
            state_segment = self.state_traj[start_iter:end_iter+1, :]
            input_segment = self.input_traj[start_iter:end_iter, :] if start_iter < end_iter else None
            
            segments.append({
                'name': self.waypoints[i][1],
                'target': self.waypoints[i][0],
                'target_type': self.waypoints[i][4],
                'state': state_segment,
                'input': input_segment,
                'error': self.waypoint_errors[i] if i < len(self.waypoint_errors) else None
            })
        
        return segments

def build_waypoint_mission_simulation(config: settings.Config, waypoints: List[Tuple[jnp.array, str, int, float, str]], 
                                      max_velocity: float = 0.5, velocity_penalty: float = 25.0) -> WaypointMissionSimulation:
    """
    Build waypoint mission simulation with speed control
    
    Args:
        config: MPC configuration
        waypoints: List of waypoints with target type
        max_velocity: Maximum desired velocity in m/s (default 0.5)
        velocity_penalty: Weight for velocity penalty (default 25.0)
    """
    
    q_init = jnp.array([0., 0., 0.8, 1., 0., 0., 0., 0., 0.], dtype=jnp.float32)
    
    # Create objective with speed control
    objective = LoadAwareObjective(max_velocity=max_velocity, velocity_penalty_weight=velocity_penalty)
    
    system, x_init, state_init = build_model_from_config(
        config.solver_dynamics, config, quadrotor_dynamics)
    
    rollout_generator = RolloutGenerator(system, objective, config)
    sampler = MPPISampler(config)
    gains = MPPIGain(config)
    
    visualizer_params = {ROBOT_SCENE_PATH_KEY: config.robot.robot_scene_path}
    
    sim = WaypointMissionSimulation(
        state_init, system, rollout_generator, sampler, gains,
        waypoints, config, visualizer_params, obstacles=False
    )
    
    # Warm up JIT
    first_reference = sim.get_current_reference()
    input_sequence = sim.controller.command(x_init, first_reference, False).block_until_ready()
    
    return sim

def plot_waypoint_results(sim: WaypointMissionSimulation):
    """Plot the results of the waypoint mission"""
    segments = sim.get_waypoint_trajectories()
    dt = 0.015
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create 3D trajectory plot
    ax_3d = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot waypoint markers
    for wp in sim.waypoints:
        ax_3d.scatter(*wp[0], s=100, marker='o', alpha=0.7, label=wp[1])
    
    # Plot bucket trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        bucket_traj = jnp.array([
            compute_bucket_position(segment['state'][j, 0:3], segment['state'][j, 7], segment['state'][j, 8]) 
            for j in range(segment['state'].shape[0])
        ])
        ax_3d.plot(bucket_traj[:, 0], bucket_traj[:, 1], bucket_traj[:, 2], 
                  color=colors[i], linewidth=2, alpha=0.8)
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Bucket Trajectory')
    ax_3d.legend(loc='upper right', fontsize=8)
    ax_3d.grid(True, alpha=0.3)
    
    # Height tracking over time
    ax_height = fig.add_subplot(2, 3, 2)
    time_offset = 0
    for i, segment in enumerate(segments):
        bucket_traj = jnp.array([
            compute_bucket_position(segment['state'][j, 0:3], segment['state'][j, 7], segment['state'][j, 8]) 
            for j in range(segment['state'].shape[0])
        ])
        time_array = dt * jnp.arange(bucket_traj.shape[0]) + time_offset
        ax_height.plot(time_array, bucket_traj[:, 2], color=colors[i], 
                      linewidth=2, label=segment['name'])
        time_offset = time_array[-1]
    
    # Add waypoint height markers
    for wp_time, wp in zip(sim.waypoint_completion_times, sim.waypoints):
        ax_height.axhline(y=wp[0][2], color='gray', linestyle=':', alpha=0.5)
        
    ax_height.set_xlabel('Time (s)')
    ax_height.set_ylabel('Height (m)')
    ax_height.set_title('Bucket Height vs Time')
    ax_height.legend(fontsize=8)
    ax_height.grid(True, alpha=0.3)
    
    # Position error over time
    ax_error = fig.add_subplot(2, 3, 3)
    time_offset = 0
    for i, segment in enumerate(segments):
        bucket_traj = jnp.array([
            compute_bucket_position(segment['state'][j, 0:3], segment['state'][j, 7], segment['state'][j, 8]) 
            for j in range(segment['state'].shape[0])
        ])
        errors = jnp.array([jnp.linalg.norm(bucket_traj[j] - segment['target']) 
                           for j in range(len(bucket_traj))])
        time_array = dt * jnp.arange(len(errors)) + time_offset
        ax_error.plot(time_array, errors, color=colors[i], linewidth=2, label=segment['name'])
        time_offset = time_array[-1]
    
    ax_error.set_xlabel('Time (s)')
    ax_error.set_ylabel('Position Error (m)')
    ax_error.set_title('Bucket Position Error')
    ax_error.legend(fontsize=8)
    ax_error.grid(True, alpha=0.3)
    ax_error.set_ylim(bottom=0)
    
    # Swing angles
    ax_alpha = fig.add_subplot(2, 3, 4)
    time_offset = 0
    for i, segment in enumerate(segments):
        time_array = dt * jnp.arange(segment['state'].shape[0]) + time_offset
        ax_alpha.plot(time_array, segment['state'][:, 7] * 180/jnp.pi, 
                     color=colors[i], linewidth=2, label=segment['name'])
        time_offset = time_array[-1]
    
    ax_alpha.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_alpha.set_xlabel('Time (s)')
    ax_alpha.set_ylabel('Angle (degrees)')
    ax_alpha.set_title('Swing Angle α (X-Z Plane)')
    ax_alpha.legend(fontsize=8)
    ax_alpha.grid(True, alpha=0.3)
    
    ax_beta = fig.add_subplot(2, 3, 5)
    time_offset = 0
    for i, segment in enumerate(segments):
        time_array = dt * jnp.arange(segment['state'].shape[0]) + time_offset
        ax_beta.plot(time_array, segment['state'][:, 8] * 180/jnp.pi, 
                    color=colors[i], linewidth=2, label=segment['name'])
        time_offset = time_array[-1]
    
    ax_beta.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_beta.set_xlabel('Time (s)')
    ax_beta.set_ylabel('Angle (degrees)')
    ax_beta.set_title('Swing Angle β (Y-Z Plane)')
    ax_beta.legend(fontsize=8)
    ax_beta.grid(True, alpha=0.3)
    
    # Velocity magnitude
    ax_vel = fig.add_subplot(2, 3, 6)
    time_offset = 0
    for i, segment in enumerate(segments):
        vel_mag = jnp.linalg.norm(segment['state'][:, 9:12], axis=1)
        time_array = dt * jnp.arange(len(vel_mag)) + time_offset
        ax_vel.plot(time_array, vel_mag, color=colors[i], linewidth=2, label=segment['name'])
        time_offset = time_array[-1]
    
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('Drone Velocity Magnitude')
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, alpha=0.3)
    
    fig.suptitle('Waypoint Mission Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def export_for_crazyflie(sim, filename: str = "crazyflie_trajectory.py"):
    """
    Export trajectory as Python script for Crazyflie drone
    """
    segments = sim.get_waypoint_trajectories()
    
    script = '''#!/usr/bin/env python3
"""
Auto-generated Crazyflie trajectory from simulation
Generated at: {timestamp}

Note: This trajectory accounts for drone vs bucket positioning.
      Some waypoints target the drone position directly,
      while others target where the bucket should be.
"""

import time
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

# Crazyflie URI - update this for your drone
URI = 'radio://0/80/2M/E7E7E7E7E7'
PENDULUM_LENGTH = 0.25  # meters

def fly_trajectory(scf):
    """Execute the simulated trajectory"""
    with MotionCommander(scf, default_height=0.8) as mc:
        print("Taking off...")
        time.sleep(2)
        
'''.format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Add waypoint movements
    for i, segment in enumerate(segments):
        target = segment['target']
        target_type = segment['target_type']
        duration = len(segment['state']) * 0.015  # Convert iterations to seconds
        
        # Get actual drone position to fly to
        if target_type == 'drone':
            # Target is for drone position directly
            drone_target = target
            comment = f"# Drone flies to position"
        else:  # target_type == 'bucket'
            # Target is for bucket, so drone needs to be above it
            # For simplified real-world testing, we approximate drone position
            drone_target = target.copy()
            if target[2] < 0.3:  # If bucket needs to be low (water collection)
                drone_target = jnp.array([target[0], target[1], target[2] + PENDULUM_LENGTH])
            comment = f"# Drone positions to place bucket at target"
        
        # Calculate relative movement from previous position
        if i == 0:
            # First waypoint - move from start position (0, 0, 0.8)
            dx = float(drone_target[0] - 0.0)
            dy = float(drone_target[1] - 0.0)
            dz = float(drone_target[2] - 0.8)
        else:
            prev_segment = segments[i-1]
            prev_target = prev_segment['target']
            prev_type = prev_segment['target_type']
            
            if prev_type == 'drone':
                prev_drone = prev_target
            else:
                prev_drone = prev_target.copy()
                if prev_target[2] < 0.3:
                    prev_drone = jnp.array([prev_target[0], prev_target[1], prev_target[2] + PENDULUM_LENGTH])
            
            dx = float(drone_target[0] - prev_drone[0])
            dy = float(drone_target[1] - prev_drone[1])
            dz = float(drone_target[2] - prev_drone[2])
        
        # Add appropriate movement command
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            # Vertical movement only
            if abs(dz) > 0.01:
                script += f'''        # Waypoint {i+1}: {segment['name']} ({target_type} target)
        {comment}
        print("Flying to {segment['name']}: ({drone_target[0]:.2f}, {drone_target[1]:.2f}, {drone_target[2]:.2f})")
        mc.move_distance(0, 0, {dz:.2f}, velocity=0.2)
        time.sleep({duration:.1f})
        
'''
        else:
            # Horizontal or combined movement
            script += f'''        # Waypoint {i+1}: {segment['name']} ({target_type} target)
        {comment}
        print("Flying to {segment['name']}: ({drone_target[0]:.2f}, {drone_target[1]:.2f}, {drone_target[2]:.2f})")
        mc.move_distance({dx:.2f}, {dy:.2f}, {dz:.2f}, velocity=0.3)
        time.sleep({duration:.1f})
        
'''
    
    script += '''        # Land
        print("Landing...")
        mc.land()
        
        print("Mission completed!")

def main():
    """Main function to connect and fly"""
    # Initialize drivers
    cflib.crtp.init_drivers()
    
    print(f"Connecting to Crazyflie at {URI}...")
    
    with SyncCrazyflie(URI, cf=None) as scf:
        print("Connected! Starting trajectory...")
        fly_trajectory(scf)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\\nFlight interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
'''
    
    with open(filename, 'w') as f:
        f.write(script)
    
    print(f"\n✅ Crazyflie trajectory script exported to: {filename}")
    print(f"📋 To fly this trajectory:")
    print(f"   1. Copy {filename} to your Crazyflie computer")
    print(f"   2. Update the URI in the script if needed")
    print(f"   3. Run: python3 {filename}")
    
    # Also export a detailed waypoint file for reference
    waypoints_filename = filename.replace('.py', '_waypoints.txt')
    with open(waypoints_filename, 'w') as f:
        f.write("Waypoint Sequence with Target Types:\n")
        f.write("-" * 50 + "\n")
        for i, segment in enumerate(segments):
            target = segment['target']
            target_type = segment['target_type']
            
            # Calculate where drone and bucket will actually be
            final_state = segment['state'][-1]
            drone_pos = final_state[0:3]
            bucket_pos = compute_bucket_position(drone_pos, final_state[7], final_state[8])
            
            f.write(f"\n{i+1}. {segment['name']}:\n")
            f.write(f"   Target Type: {target_type.upper()}\n")
            f.write(f"   Target Position: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})\n")
            f.write(f"   Drone Achieved: ({drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f})\n")
            f.write(f"   Bucket Achieved: ({bucket_pos[0]:.3f}, {bucket_pos[1]:.3f}, {bucket_pos[2]:.3f})\n")
    
    print(f"📄 Detailed waypoint reference saved to: {waypoints_filename}")

def run_waypoint_water_mission(speed_mode='normal'):
    """
    Run the waypoint-based water collection and lift mission
    
    Args:
        speed_mode: 'slow', 'normal', or 'fast' - controls drone speed
    """
    print("="*60)
    print("WAYPOINT-BASED DRONE WATER MISSION SIMULATION")
    print("="*60)
    
    # Speed configurations with target types
    speed_configs = {
        'slow': {
            'max_velocity': 0.3,  # m/s
            'velocity_penalty': 40.0,
            'waypoints': [
                (jnp.array([0.5, 0.5, 0.8], dtype=jnp.float32), "Above Water", 300, 1.0, 'drone'),
                (jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32), "Water Target", 300, 1.0, 'bucket'),
                (jnp.array([0.5, 0.5, 0.8], dtype=jnp.float32), "Above Water (Return)", 400, 12.1, 'drone'),
                (jnp.array([0.8, 0.3, 0.3], dtype=jnp.float32), "Lift Target", 400, 12.1, 'bucket'),
            ]
        },
        'normal': {
            'max_velocity': 0.5,  # m/s
            'velocity_penalty': 25.0,
            'waypoints': [
                (jnp.array([0.5, 0.5, 0.8], dtype=jnp.float32), "Above Water", 200, 1.0, 'drone'),
                (jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32), "Water Target", 200, 1.0, 'bucket'),
                (jnp.array([0.5, 0.5, 0.8], dtype=jnp.float32), "Above Water (Return)", 250, 12.1, 'drone'),
                (jnp.array([0.8, 0.3, 0.3], dtype=jnp.float32), "Lift Target", 250, 12.1, 'bucket'),
            ]
        },
        'fast': {
            'max_velocity': 1.0,  # m/s
            'velocity_penalty': 8.0,
            'waypoints': WAYPOINTS  # Uses global WAYPOINTS with target types
        }
    }
    
    # Get speed configuration
    speed_config = speed_configs.get(speed_mode, speed_configs['normal'])
    
    print(f"\nSpeed Mode: {speed_mode.upper()}")
    print(f"  Max Velocity: {speed_config['max_velocity']} m/s")
    print(f"  Velocity Penalty: {speed_config['velocity_penalty']}")
    
    print("\nMission Trajectory:")
    for i, wp in enumerate(speed_config['waypoints']):
        pos, name, iters, mass, target_type = wp
        target_str = "🚁 Drone" if target_type == 'drone' else "🪣 Bucket"
        print(f"  {i+1}. {name}: {pos} ({target_str} target, {iters} iters)")
    
    print("\nFeatures:")
    print("  - Dual targeting: drone OR bucket positioning")
    print("  - Speed-controlled waypoint navigation")
    print("  - Load-aware MPC controller")
    print("  - Automatic mass adjustment after water collection")
    print("="*60 + "\n")
    
    # Create robot configuration
    q_init = jnp.array([0., 0., 0.8, 1., 0., 0., 0., 0., 0.], dtype=jnp.float32)
    robot_config = create_robot_config(q_init)
    
    # Create MPC configuration
    uav_mass = 0.027  # kg
    config = create_mpc_config(robot_config, uav_mass)
    
    # Set total simulation iterations
    total_iters = sum(wp[2] for wp in speed_config['waypoints'])
    config.sim_iterations = total_iters
    
    try:
        # Build and run simulation with speed control
        sim = build_waypoint_mission_simulation(
            config, 
            speed_config['waypoints'],
            max_velocity=speed_config['max_velocity'],
            velocity_penalty=speed_config['velocity_penalty']
        )
        sim.simulate()
        
        # Print summary
        print("\n" + "="*60)
        print("MISSION RESULTS SUMMARY")
        print("="*60)
        
        segments = sim.get_waypoint_trajectories()
        max_velocities = []
        
        for i, segment in enumerate(segments):
            print(f"\nWaypoint {i+1}: {segment['name']} ({segment['target_type']} target)")
            print(f"  Target: {segment['target']}")
            
            if segment['error'] is not None:
                print(f"  Final Error: {segment['error']:.4f} m")
            
            # Calculate actual positions achieved
            final_state = segment['state'][-1]
            drone_pos = final_state[0:3]
            bucket_pos = compute_bucket_position(drone_pos, final_state[7], final_state[8])
            print(f"  Drone reached: ({drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f})")
            print(f"  Bucket reached: ({bucket_pos[0]:.3f}, {bucket_pos[1]:.3f}, {bucket_pos[2]:.3f})")
            
            # Calculate max velocity for this segment
            velocities = jnp.linalg.norm(segment['state'][:, 9:12], axis=1)
            max_vel = float(jnp.max(velocities))
            avg_vel = float(jnp.mean(velocities))
            max_velocities.append(max_vel)
            print(f"  Max Velocity: {max_vel:.3f} m/s")
            print(f"  Avg Velocity: {avg_vel:.3f} m/s")
        
        # Calculate overall metrics
        max_error = max(sim.waypoint_errors)
        avg_error = np.mean(sim.waypoint_errors)
        total_time = sim.waypoint_completion_times[-1]
        overall_max_vel = max(max_velocities)
        
        print(f"\nOverall Performance:")
        print(f"  Maximum Error: {max_error:.4f} m")
        print(f"  Average Error: {avg_error:.4f} m")
        print(f"  Maximum Velocity: {overall_max_vel:.3f} m/s")
        print(f"  Total Mission Time: {total_time:.2f} s")
        
        print("="*60)
        
        # Generate plots
        plot_waypoint_results(sim)
        
        return sim
        
    except Exception as e:
        print(f"❌ Mission failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the waypoint-based water mission
    # Options: 'slow', 'normal', 'fast'
    sim = run_waypoint_water_mission(speed_mode='slow')  # Change this to control speed
    
    if sim is not None:
        print("\n🎉 Mission completed successfully!")
        print("📊 Results plotted")
        
        # Export trajectory for Crazyflie
        export_for_crazyflie(sim, "crazyflie_mission.py")
        
        print("\n🚁 Ready for real-world testing with Crazyflie!")