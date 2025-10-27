import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import LineCollection
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class DifferentialDriveCar:
    """
    A differential drive robot with realistic kinematics.
    
    Attributes:
        x, y: Position in meters
        theta: Orientation in radians
        v: Linear velocity in m/s
        w: Angular velocity in rad/s
        wheel_base: Distance between wheels in meters
        max_velocity: Maximum linear velocity in m/s
        max_angular_velocity: Maximum angular velocity in rad/s
    """
    
    def __init__(self, initial_x=0.0, initial_y=0.0, initial_theta=0.0, 
                 wheel_base=0.5, max_velocity=2.0, max_angular_velocity=2.0):
        self.x = initial_x
        self.y = initial_y
        self.theta = initial_theta
        self.v = 0.0
        self.w = 0.0
        self.wheel_base = wheel_base
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        
        # History tracking
        self.trajectory = [(self.x, self.y)]
        self.theta_history = [self.theta]
        self.velocity_history = [(self.v, self.w)]
        self.time_history = [0.0]
        
    def update_state(self, v, w, dt):
        """
        Update car state using differential drive kinematics.
        
        Args:
            v: Desired linear velocity (m/s)
            w: Desired angular velocity (rad/s)
            dt: Time step (seconds)
        """
        # Clamp velocities to limits
        v = np.clip(v, -self.max_velocity, self.max_velocity)
        w = np.clip(w, -self.max_angular_velocity, self.max_angular_velocity)
        
        self.v = v
        self.w = w
        
        # Update position using exact integration for circular arc
        if abs(w) > 1e-6:
            radius = v / w
            self.x += radius * (math.sin(self.theta + w * dt) - math.sin(self.theta))
            self.y += radius * (-math.cos(self.theta + w * dt) + math.cos(self.theta))
            self.theta += w * dt
        else:
            # Straight line motion
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        
        # Normalize angle to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Record history
        current_time = self.time_history[-1] + dt if self.time_history else dt
        self.trajectory.append((self.x, self.y))
        self.theta_history.append(self.theta)
        self.velocity_history.append((self.v, self.w))
        self.time_history.append(current_time)
    
    def get_state(self):
        """Return current state as a dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'v': self.v,
            'w': self.w,
            't': self.time_history[-1] if self.time_history else 0.0
        }
    
    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Reset the car to initial state."""
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.w = 0.0
        self.trajectory = [(self.x, self.y)]
        self.theta_history = [self.theta]
        self.velocity_history = [(self.v, self.w)]
        self.time_history = [0.0]


class TrajectoryGenerator:
    """Generate time-parameterized reference trajectories for testing."""
    
    @staticmethod
    def circle(center=(0, 0), radius=5.0, angular_velocity=0.3, duration=20.0, dt=0.1):
        """
        Generate a circular trajectory with time stamps.
        
        Args:
            center: Circle center (x, y)
            radius: Circle radius in meters
            angular_velocity: Angular velocity in rad/s
            duration: Total duration in seconds
            dt: Time step in seconds
            
        Returns:
            List of (t, x, y, theta, v, w) tuples
        """
        trajectory = []
        t = 0.0
        
        while t <= duration:
            angle = angular_velocity * t
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            theta = angle + np.pi / 2  # Tangent to circle
            v = radius * angular_velocity
            w = angular_velocity
            
            trajectory.append((t, x, y, theta, v, w))
            t += dt
        
        return trajectory
    
    @staticmethod
    def figure_eight(center=(0, 0), radius=5.0, period=20.0, duration=40.0, dt=0.1):
        """
        Generate a figure-eight trajectory with time stamps.
        
        Args:
            center: Center point (x, y)
            radius: Scale of the figure eight
            period: Time to complete one full figure eight
            duration: Total duration in seconds
            dt: Time step in seconds
            
        Returns:
            List of (t, x, y, theta, v, w) tuples
        """
        trajectory = []
        t = 0.0
        
        while t <= duration:
            omega = 2 * np.pi / period
            param = omega * t
            
            x = center[0] + radius * np.sin(param)
            y = center[1] + radius * np.sin(param) * np.cos(param)
            
            # Compute derivatives for velocity and orientation
            dx_dt = radius * omega * np.cos(param)
            dy_dt = radius * omega * (np.cos(2 * param))
            
            v = np.sqrt(dx_dt**2 + dy_dt**2)
            theta = np.arctan2(dy_dt, dx_dt)
            
            # Compute angular velocity (derivative of theta)
            # Numerical approximation for simplicity
            if trajectory:
                prev_theta = trajectory[-1][3]
                w = (theta - prev_theta) / dt
                # Normalize angular difference
                w = np.arctan2(np.sin(w * dt), np.cos(w * dt)) / dt
            else:
                w = 0.0
            
            trajectory.append((t, x, y, theta, v, w))
            t += dt
        
        return trajectory
    
    @staticmethod
    def straight_line(start=(0, 0), end=(10, 0), velocity=1.0, dt=0.1):
        """
        Generate a straight line trajectory with constant velocity.
        
        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            velocity: Linear velocity in m/s
            dt: Time step in seconds
            
        Returns:
            List of (t, x, y, theta, v, w) tuples
        """
        distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        duration = distance / velocity
        theta = np.arctan2(end[1] - start[1], end[0] - start[0])
        
        trajectory = []
        t = 0.0
        
        while t <= duration:
            progress = min(t * velocity / distance, 1.0)
            x = start[0] + (end[0] - start[0]) * progress
            y = start[1] + (end[1] - start[1]) * progress
            
            trajectory.append((t, x, y, theta, velocity, 0.0))
            t += dt
        
        return trajectory
    
    @staticmethod
    def s_curve(length=15.0, amplitude=3.0, velocity=1.0, duration=15.0, dt=0.1):
        """
        Generate an S-curve trajectory.
        
        Args:
            length: Total length in x-direction
            amplitude: Wave amplitude
            velocity: Average forward velocity
            duration: Total duration in seconds
            dt: Time step in seconds
            
        Returns:
            List of (t, x, y, theta, v, w) tuples
        """
        trajectory = []
        t = 0.0
        
        while t <= duration:
            progress = t / duration
            x = length * progress
            y = amplitude * np.sin(2 * np.pi * progress)
            
            # Compute derivatives
            dx_dt = length / duration
            dy_dt = amplitude * (2 * np.pi / duration) * np.cos(2 * np.pi * progress)
            
            v_inst = np.sqrt(dx_dt**2 + dy_dt**2)
            theta = np.arctan2(dy_dt, dx_dt)
            
            # Compute angular velocity
            if trajectory:
                prev_theta = trajectory[-1][3]
                dtheta = theta - prev_theta
                # Normalize to [-pi, pi]
                dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                w = dtheta / dt
            else:
                w = 0.0
            
            trajectory.append((t, x, y, theta, v_inst, w))
            t += dt
        
        return trajectory
    
    @staticmethod
    def parking_maneuver(start=(0, 0), parking_spot=(10, 3), dt=0.1):
        """
        Generate a parallel parking maneuver trajectory.
        
        Args:
            start: Starting position (x, y)
            parking_spot: Target parking position (x, y)
            dt: Time step in seconds
            
        Returns:
            List of (t, x, y, theta, v, w) tuples
        """
        trajectory = []
        t = 0.0
        
        # Phase 1: Move forward
        approach_distance = parking_spot[0] - 3.0
        for i in range(int(approach_distance / (0.5 * dt))):
            x = start[0] + 0.5 * t
            y = start[1]
            trajectory.append((t, x, y, 0.0, 0.5, 0.0))
            t += dt
        
        # Phase 2: Arc backward into spot
        arc_steps = 30
        for i in range(arc_steps):
            angle = (i / arc_steps) * (np.pi / 3)
            radius = 3.0
            x = approach_distance - radius * np.sin(angle)
            y = start[1] + radius * (1 - np.cos(angle))
            theta = -angle
            trajectory.append((t, x, y, theta, -0.3, 0.3))
            t += dt
        
        # Phase 3: Straighten
        final_x = trajectory[-1][1]
        final_y = trajectory[-1][2]
        for i in range(10):
            x = final_x + 0.1 * i * dt
            trajectory.append((t, x, parking_spot[1], -np.pi/6, 0.1, 0.0))
            t += dt
        
        return trajectory
    
    @staticmethod
    def custom_waypoints(waypoints, velocities=None, dt=0.1):
        """
        Generate trajectory from waypoints with smooth interpolation.
        
        Args:
            waypoints: List of (x, y) tuples
            velocities: List of velocities for each segment (or None for constant)
            dt: Time step in seconds
            
        Returns:
            List of (t, x, y, theta, v, w) tuples
        """
        if velocities is None:
            velocities = [1.0] * (len(waypoints) - 1)
        
        trajectory = []
        t = 0.0
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            v = velocities[i]
            
            distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            segment_duration = distance / v
            theta = np.arctan2(end[1] - start[1], end[0] - start[0])
            
            steps = int(segment_duration / dt)
            for step in range(steps):
                progress = step / steps
                x = start[0] + (end[0] - start[0]) * progress
                y = start[1] + (end[1] - start[1]) * progress
                
                # Compute angular velocity at waypoint transitions
                w = 0.0
                if step == 0 and i > 0:
                    prev_theta = trajectory[-1][3]
                    dtheta = theta - prev_theta
                    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                    w = dtheta / dt if dt > 0 else 0.0
                
                trajectory.append((t, x, y, theta, v, w))
                t += dt
        
        return trajectory


class SceneRenderer:
    """
    Visualize the car, reference trajectory, and actual trajectory with enhanced graphics.
    Includes MPC predicted trajectory visualization.
    """
    
    def __init__(self, figsize=(14, 11)):
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('Differential Drive Car Simulation - Trajectory Tracking', 
                         fontsize=14, fontweight='bold')
        
        # Main plot for car and trajectory
        self.ax_main = self.axes[0, 0]
        self.ax_main.set_aspect('equal', 'box')
        self.ax_main.set_xlabel('X Position (m)', fontsize=10)
        self.ax_main.set_ylabel('Y Position (m)', fontsize=10)
        self.ax_main.set_title('Top View - Trajectory Tracking')
        self.ax_main.grid(True, alpha=0.3)
        
        # Velocity plot
        self.ax_velocity = self.axes[0, 1]
        self.ax_velocity.set_xlabel('Time (s)', fontsize=10)
        self.ax_velocity.set_ylabel('Velocity', fontsize=10)
        self.ax_velocity.set_title('Velocity Tracking')
        self.ax_velocity.grid(True, alpha=0.3)
        
        # Position error plot
        self.ax_error = self.axes[1, 0]
        self.ax_error.set_xlabel('Time (s)', fontsize=10)
        self.ax_error.set_ylabel('Error (m)', fontsize=10)
        self.ax_error.set_title('Position & Orientation Error')
        self.ax_error.grid(True, alpha=0.3)
        
        # X-Y error plot
        self.ax_xy_error = self.axes[1, 1]
        self.ax_xy_error.set_xlabel('Time (s)', fontsize=10)
        self.ax_xy_error.set_ylabel('Error (m)', fontsize=10)
        self.ax_xy_error.set_title('X-Y Position Errors')
        self.ax_xy_error.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def draw_car(self, car, color='red'):
        """Draw a detailed car representation."""
        length = 0.8
        width = 0.5
        
        corners = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ])
        
        cos_theta = math.cos(car.theta)
        sin_theta = math.sin(car.theta)
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        
        rotated_corners = corners @ R.T
        global_corners = rotated_corners + np.array([car.x, car.y])
        
        car_body = plt.Polygon(global_corners, facecolor=color, edgecolor='black', 
                               linewidth=2, alpha=0.7, zorder=5)
        self.ax_main.add_patch(car_body)
        
        arrow_length = length * 0.8
        dx = arrow_length * cos_theta
        dy = arrow_length * sin_theta
        self.ax_main.arrow(car.x, car.y, dx, dy, 
                          head_width=0.3, head_length=0.2, 
                          fc='yellow', ec='black', linewidth=1.5, zorder=6)
        
        wheel_positions = [
            (length/3, width/2), (length/3, -width/2),
            (-length/3, width/2), (-length/3, -width/2)
        ]
        
        for wx, wy in wheel_positions:
            wheel_local = np.array([wx, wy])
            wheel_global = wheel_local @ R.T + np.array([car.x, car.y])
            wheel = plt.Circle(wheel_global, 0.08, color='black', zorder=6)
            self.ax_main.add_patch(wheel)
    
    def render(self, car, reference_trajectory=None, show_trajectory=True, mpc_prediction=None):
        """
        Render the complete scene with all visualizations.
        
        Args:
            car: DifferentialDriveCar instance
            reference_trajectory: Reference trajectory (list of tuples)
            show_trajectory: Whether to show actual trajectory
            mpc_prediction: MPC predicted trajectory (list of [x, y, theta] arrays)
        """
        for ax in self.axes.flat:
            ax.cla()
            ax.grid(True, alpha=0.3)
        
        # --- Main view ---
        self.ax_main.set_aspect('equal', 'box')
        self.ax_main.set_xlabel('X Position (m)', fontsize=10)
        self.ax_main.set_ylabel('Y Position (m)', fontsize=10)
        self.ax_main.set_title('Top View - Trajectory Tracking')
        
        # Plot reference trajectory
        if reference_trajectory:
            ref_t, ref_x, ref_y, ref_theta, ref_v, ref_w = zip(*reference_trajectory)
            self.ax_main.plot(ref_x, ref_y, 'b-', linewidth=3, 
                            label='Reference Trajectory', alpha=0.4)
            self.ax_main.plot(ref_x[0], ref_y[0], 'go', markersize=12, 
                            label='Start', zorder=4)
            self.ax_main.plot(ref_x[-1], ref_y[-1], 'rs', markersize=12, 
                            label='Goal', zorder=4)
            
            # Draw time markers along reference
            marker_interval = max(1, len(ref_x) // 10)
            for i in range(0, len(ref_x), marker_interval):
                self.ax_main.plot(ref_x[i], ref_y[i], 'b.', markersize=8, alpha=0.6)
                self.ax_main.text(ref_x[i], ref_y[i], f' t={ref_t[i]:.1f}s', 
                                fontsize=7, alpha=0.7)
        
        # Plot MPC predicted trajectory
        if mpc_prediction and len(mpc_prediction) > 1:
            pred_x = [state[0] for state in mpc_prediction]
            pred_y = [state[1] for state in mpc_prediction]
            self.ax_main.plot(pred_x, pred_y, 'mo-', linewidth=2, 
                            markersize=4, label='MPC Prediction', alpha=0.8, zorder=4)
            
            # Draw orientation arrows for predicted states
            for i in range(0, len(mpc_prediction), max(1, len(mpc_prediction)//5)):
                state = mpc_prediction[i]
                arrow_len = 0.4
                dx = arrow_len * np.cos(state[2])
                dy = arrow_len * np.sin(state[2])
                self.ax_main.arrow(state[0], state[1], dx, dy,
                                 head_width=0.15, head_length=0.1,
                                 fc='magenta', ec='magenta', alpha=0.6, zorder=4)
        
        # Plot actual trajectory
        if show_trajectory and len(car.trajectory) > 1:
            points = np.array(car.trajectory)
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
            
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(segments)))
            lc = LineCollection(segments, colors=colors, linewidth=2.5, 
                              alpha=0.9, label='Actual Trajectory', zorder=3)
            self.ax_main.add_collection(lc)
        
        # Draw car
        self.draw_car(car)
        
        # Set dynamic limits
        if reference_trajectory:
            _, ref_x, ref_y, _, _, _ = zip(*reference_trajectory)
            all_x = list(ref_x) + [p[0] for p in car.trajectory]
            all_y = list(ref_y) + [p[1] for p in car.trajectory]
        else:
            all_x = [p[0] for p in car.trajectory]
            all_y = [p[1] for p in car.trajectory]
        
        if mpc_prediction:
            all_x.extend([state[0] for state in mpc_prediction])
            all_y.extend([state[1] for state in mpc_prediction])
        
        padding = 2.0
        self.ax_main.set_xlim(min(all_x) - padding, max(all_x) + padding)
        self.ax_main.set_ylim(min(all_y) - padding, max(all_y) + padding)
        self.ax_main.legend(loc='upper right', fontsize=8)
        
        # --- Velocity tracking plot ---
        if reference_trajectory and len(car.velocity_history) > 1:
            ref_t, _, _, _, ref_v, ref_w = zip(*reference_trajectory)
            actual_v, actual_w = zip(*car.velocity_history)
            times = car.time_history
            
            self.ax_velocity.plot(ref_t, ref_v, 'b--', linewidth=2, 
                                label='Ref Linear v (m/s)', alpha=0.7)
            self.ax_velocity.plot(times, actual_v, 'b-', linewidth=2, 
                                label='Actual Linear v')
            self.ax_velocity.plot(ref_t, ref_w, 'r--', linewidth=2, 
                                label='Ref Angular ω (rad/s)', alpha=0.7)
            self.ax_velocity.plot(times, actual_w, 'r-', linewidth=2, 
                                label='Actual Angular ω')
            self.ax_velocity.legend(fontsize=7, loc='upper right')
            self.ax_velocity.set_xlabel('Time (s)', fontsize=10)
            self.ax_velocity.set_title('Velocity Tracking')
        
        # --- Position and orientation error plot ---
        if reference_trajectory and len(car.trajectory) > 1:
            ref_times, ref_xs, ref_ys, ref_thetas, _, _ = zip(*reference_trajectory)
            
            position_errors = []
            orientation_errors = []
            x_errors = []
            y_errors = []
            error_times = []
            
            for i, (actual_t, (actual_x, actual_y)) in enumerate(zip(car.time_history, car.trajectory)):
                # Find closest reference point by time
                idx = min(range(len(ref_times)), 
                         key=lambda j: abs(ref_times[j] - actual_t))
                
                ref_x, ref_y = ref_xs[idx], ref_ys[idx]
                ref_theta = ref_thetas[idx]
                
                # Position error
                pos_error = np.sqrt((actual_x - ref_x)**2 + (actual_y - ref_y)**2)
                position_errors.append(pos_error)
                x_errors.append(abs(actual_x - ref_x))
                y_errors.append(abs(actual_y - ref_y))
                
                # Orientation error
                if i < len(car.theta_history):
                    actual_theta = car.theta_history[i]
                    theta_error = abs(np.arctan2(np.sin(actual_theta - ref_theta), 
                                                 np.cos(actual_theta - ref_theta)))
                    orientation_errors.append(theta_error)
                
                error_times.append(actual_t)
            
            # Plot position and orientation errors
            self.ax_error.plot(error_times, position_errors, 'g-', linewidth=2, 
                             label=f'Position Error (avg: {np.mean(position_errors):.3f}m)')
            self.ax_error.plot(error_times, orientation_errors, 'm-', linewidth=2, 
                             label=f'Orientation Error (avg: {np.mean(orientation_errors):.3f}rad)')
            self.ax_error.legend(fontsize=8)
            self.ax_error.set_title('Position & Orientation Error')
            
            # Plot X-Y errors separately
            self.ax_xy_error.plot(error_times, x_errors, 'b-', linewidth=2, 
                                label=f'X Error (avg: {np.mean(x_errors):.3f}m)')
            self.ax_xy_error.plot(error_times, y_errors, 'r-', linewidth=2, 
                                label=f'Y Error (avg: {np.mean(y_errors):.3f}m)')
            self.ax_xy_error.legend(fontsize=8)
            self.ax_xy_error.set_title('X-Y Position Errors')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def show(self):
        """Display the final plot."""
        plt.show()
        
        car_body = plt.Polygon(global_corners, facecolor=color, edgecolor='black', 
                               linewidth=2, alpha=0.7, zorder=5)
        self.ax_main.add_patch(car_body)
        
        arrow_length = length * 0.8
        dx = arrow_length * cos_theta
        dy = arrow_length * sin_theta
        self.ax_main.arrow(car.x, car.y, dx, dy, 
                          head_width=0.3, head_length=0.2, 
                          fc='yellow', ec='black', linewidth=1.5, zorder=6)
        
        wheel_positions = [
            (length/3, width/2), (length/3, -width/2),
            (-length/3, width/2), (-length/3, -width/2)
        ]
        
        for wx, wy in wheel_positions:
            wheel_local = np.array([wx, wy])
            wheel_global = wheel_local @ R.T + np.array([car.x, car.y])
            wheel = plt.Circle(wheel_global, 0.08, color='black', zorder=6)
            self.ax_main.add_patch(wheel)
    

    def show(self):
        """Display the final plot."""
        plt.show()


class Simulator:
    """
    Main simulation interface for testing control algorithms with time-parameterized trajectories.
    """
    
    def __init__(self, car, renderer=None, dt=0.1):
        self.car = car
        self.renderer = renderer if renderer else SceneRenderer()
        self.dt = dt
        self.reference_trajectory = None
        
    def set_reference_trajectory(self, trajectory):
        """
        Set the reference trajectory for the simulation.
        
        Args:
            trajectory: List of (t, x, y, theta, v, w) tuples
        """
        self.reference_trajectory = trajectory
        
    def run(self, controller, duration=None, render_interval=1, verbose=True):
        """
        Run simulation with a given controller.
        
        Args:
            controller: Function that takes (car_state, reference_trajectory, current_time, dt) 
                       and returns (v, w), OR an MPC controller object
            duration: Simulation duration in seconds (or None to use trajectory duration)
            render_interval: Render every N steps
            verbose: Print progress
        """
        if duration is None and self.reference_trajectory:
            duration = self.reference_trajectory[-1][0]
        elif duration is None:
            duration = 10.0
        
        steps = int(duration / self.dt)
        current_time = 0.0
        
        # Check if controller is MPC
        is_mpc = isinstance(controller, MPCController)
        
        for step in range(steps):
            # Get control inputs from controller
            state = self.car.get_state()
            v, w = controller(state, self.reference_trajectory, current_time, self.dt)
            
            # Update car state
            self.car.update_state(v, w, self.dt)
            current_time += self.dt
            
            # Render periodically
            if step % render_interval == 0:
                # If MPC, pass predicted trajectory for visualization
                mpc_pred = controller.predicted_trajectory if is_mpc else None
                self.renderer.render(self.car, self.reference_trajectory, 
                                   show_trajectory=True, mpc_prediction=mpc_pred)
                
            if verbose and step % 20 == 0:
                print(f"t={current_time:.2f}s: x={self.car.x:.2f}, y={self.car.y:.2f}, "
                      f"θ={self.car.theta:.2f}, v={self.car.v:.2f}, ω={self.car.w:.2f}")
        
        # Final render
        mpc_pred = controller.predicted_trajectory if is_mpc else None
        self.renderer.render(self.car, self.reference_trajectory,
                           show_trajectory=True, mpc_prediction=mpc_pred)
        if verbose:
            print(f"\nSimulation complete! Final position: ({self.car.x:.2f}, {self.car.y:.2f})")


class MPCController:
    """
    Model Predictive Control (MPC) controller for differential drive robot.
    
    At each time step:
    1. Query reference trajectory for the prediction horizon
    2. Construct quadratic cost function: J = sum(||x - x_ref||²_Q + ||u||²_R)
    3. Define constraints (state bounds, control bounds, dynamics, terminal constraint)
    4. Solve constrained optimization problem
    5. Apply first control input, visualize predicted trajectory
    
    Attributes:
        horizon: Prediction horizon (number of steps)
        dt: Time step for discretization
        Q: State cost matrix (weights for x, y, theta errors)
        R: Control cost matrix (weights for v, w)
        Q_terminal: Terminal state cost matrix
        max_velocity: Maximum linear velocity constraint
        max_angular_velocity: Maximum angular velocity constraint
        max_accel: Maximum acceleration constraint
        state_bounds: Bounds on states (x, y, theta)
    """
    
    def __init__(self, horizon=10, dt=0.1, 
                 Q=None, R=None, Q_terminal=None,
                 max_velocity=2.0, max_angular_velocity=2.0,
                 max_accel=1.0, state_bounds=None):
        
        self.horizon = horizon
        self.dt = dt
        
        # Cost matrices (default values if not provided)
        # Q weights: [x_error, y_error, theta_error]
        self.Q = np.diag([10.0, 10.0, 5.0]) if Q is None else Q
        
        # R weights: [v, w]
        self.R = np.diag([0.1, 0.1]) if R is None else R
        
        # Terminal cost (usually higher to ensure reaching goal)
        self.Q_terminal = np.diag([50.0, 50.0, 20.0]) if Q_terminal is None else Q_terminal
        
        # Constraints
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_accel = max_accel
        self.state_bounds = state_bounds  # Optional: (x_min, x_max, y_min, y_max)
        
        # For visualization
        self.predicted_trajectory = []
        self.predicted_controls = []
        
        # Previous control for acceleration constraint
        self.prev_v = 0.0
        self.prev_w = 0.0
        
    def get_reference_over_horizon(self, reference_trajectory, current_time):
        """
        Query reference trajectory over the prediction horizon.
        
        Returns:
            ref_states: Array of shape (horizon+1, 3) with [x, y, theta]
            ref_controls: Array of shape (horizon, 2) with [v, w]
        """
        ref_states = []
        ref_controls = []
        
        for k in range(self.horizon + 1):
            query_time = current_time + k * self.dt
            
            # Find closest reference point by time
            ref_idx = min(range(len(reference_trajectory)), 
                         key=lambda i: abs(reference_trajectory[i][0] - query_time))
            
            t, x, y, theta, v, w = reference_trajectory[ref_idx]
            ref_states.append([x, y, theta])
            
            if k < self.horizon:
                ref_controls.append([v, w])
        
        return np.array(ref_states), np.array(ref_controls)
    
    def predict_state(self, state, v, w, dt):
        """
        Predict next state using differential drive kinematics.
        
        Args:
            state: Current state [x, y, theta]
            v: Linear velocity
            w: Angular velocity
            dt: Time step
            
        Returns:
            next_state: [x_next, y_next, theta_next]
        """
        x, y, theta = state
        
        if abs(w) > 1e-6:
            # Circular motion
            radius = v / w
            x_next = x + radius * (np.sin(theta + w * dt) - np.sin(theta))
            y_next = y + radius * (-np.cos(theta + w * dt) + np.cos(theta))
            theta_next = theta + w * dt
        else:
            # Straight line motion
            x_next = x + v * np.cos(theta) * dt
            y_next = y + v * np.sin(theta) * dt
            theta_next = theta
        
        # Normalize angle
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.array([x_next, y_next, theta_next])
    
    def compute_cost(self, decision_vars, initial_state, ref_states, ref_controls):
        """
        Compute the total cost function for MPC optimization.
        
        The decision variables are structured as:
        [v_0, w_0, v_1, w_1, ..., v_{N-1}, w_{N-1}]
        where N is the horizon length.
        
        Cost = sum_{k=0}^{N-1} [(x_k - x_ref_k)^T Q (x_k - x_ref_k) + u_k^T R u_k]
               + (x_N - x_ref_N)^T Q_terminal (x_N - x_ref_N)
        """
        cost = 0.0
        state = initial_state.copy()
        
        # Extract controls from decision variables
        controls = decision_vars.reshape(self.horizon, 2)
        
        # Stage costs
        for k in range(self.horizon):
            v, w = controls[k]
            
            # State error cost
            state_error = state - ref_states[k]
            # Normalize angle error to [-pi, pi]
            state_error[2] = np.arctan2(np.sin(state_error[2]), np.cos(state_error[2]))
            cost += state_error.T @ self.Q @ state_error
            
            # Control cost
            control = np.array([v, w])
            cost += control.T @ self.R @ control
            
            # Control rate cost (smoothness) - penalize large changes
            if k > 0:
                control_rate = control - controls[k-1]
                cost += 0.5 * np.sum(control_rate**2)
            
            # Predict next state
            state = self.predict_state(state, v, w, self.dt)
        
        # Terminal cost
        terminal_error = state - ref_states[self.horizon]
        terminal_error[2] = np.arctan2(np.sin(terminal_error[2]), np.cos(terminal_error[2]))
        cost += terminal_error.T @ self.Q_terminal @ terminal_error
        
        return cost
    
    def dynamics_constraints(self, decision_vars, initial_state, ref_states):
        """
        Dynamics constraints to ensure predicted states follow the vehicle model.
        Returns a list of constraint violations (should be zero).
        """
        constraints = []
        state = initial_state.copy()
        controls = decision_vars.reshape(self.horizon, 2)
        
        for k in range(self.horizon):
            v, w = controls[k]
            next_state = self.predict_state(state, v, w, self.dt)
            state = next_state
        
        return np.array([])  # Dynamics embedded in cost, no explicit constraints needed
    
    def solve(self, current_state, reference_trajectory, current_time):
        """
        Solve the MPC optimization problem.
        
        Args:
            current_state: Current state dict {'x', 'y', 'theta', 'v', 'w'}
            reference_trajectory: Full reference trajectory
            current_time: Current simulation time
            
        Returns:
            v, w: Optimal control inputs to apply
        """
        # Get reference over horizon
        ref_states, ref_controls = self.get_reference_over_horizon(
            reference_trajectory, current_time
        )
        
        # Initial state as array
        initial_state = np.array([
            current_state['x'],
            current_state['y'],
            current_state['theta']
        ])
        
        # Initial guess: use reference controls
        initial_guess = ref_controls.flatten()
        
        # Define bounds on control inputs
        # decision_vars = [v_0, w_0, v_1, w_1, ..., v_{N-1}, w_{N-1}]
        bounds = []
        for k in range(self.horizon):
            # Linear velocity bounds
            bounds.append((-self.max_velocity, self.max_velocity))
            # Angular velocity bounds
            bounds.append((-self.max_angular_velocity, self.max_angular_velocity))
        
        # Define constraints
        constraints = []
        
        # Acceleration constraints (rate of change of control)
        def accel_constraint_v(decision_vars):
            controls = decision_vars.reshape(self.horizon, 2)
            violations = []
            prev_v = self.prev_v
            for k in range(self.horizon):
                dv = abs(controls[k, 0] - prev_v)
                violations.append(self.max_accel * self.dt - dv)
                prev_v = controls[k, 0]
            return np.array(violations)
        
        def accel_constraint_w(decision_vars):
            controls = decision_vars.reshape(self.horizon, 2)
            violations = []
            prev_w = self.prev_w
            for k in range(self.horizon):
                dw = abs(controls[k, 1] - prev_w)
                violations.append(self.max_accel * self.dt - dw)
                prev_w = controls[k, 1]
            return np.array(violations)
        
        constraints.append({
            'type': 'ineq',
            'fun': accel_constraint_v
        })
        constraints.append({
            'type': 'ineq',
            'fun': accel_constraint_w
        })
        
        # Terminal constraint (optional, soft constraint via terminal cost)
        # Could add hard constraint: ||x_N - x_ref_N|| <= epsilon
        
        # State bounds constraints (if specified)
        if self.state_bounds is not None:
            x_min, x_max, y_min, y_max = self.state_bounds
            
            def state_bounds_constraint(decision_vars):
                state = initial_state.copy()
                controls = decision_vars.reshape(self.horizon, 2)
                violations = []
                
                for k in range(self.horizon):
                    v, w = controls[k]
                    state = self.predict_state(state, v, w, self.dt)
                    
                    # x bounds
                    violations.append(state[0] - x_min)  # x >= x_min
                    violations.append(x_max - state[0])  # x <= x_max
                    
                    # y bounds
                    violations.append(state[1] - y_min)  # y >= y_min
                    violations.append(y_max - state[1])  # y <= y_max
                
                return np.array(violations)
            
            constraints.append({
                'type': 'ineq',
                'fun': state_bounds_constraint
            })
        
        # Solve optimization problem
        result = minimize(
            fun=self.compute_cost,
            x0=initial_guess,
            args=(initial_state, ref_states, ref_controls),
            method='SLSQP',  # Sequential Least Squares Programming
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
        )
        
        if not result.success:
            print(f"Warning: MPC optimization failed: {result.message}")
            # Fallback to reference controls
            optimal_controls = ref_controls
        else:
            optimal_controls = result.x.reshape(self.horizon, 2)
        
        # Store predicted trajectory for visualization
        self.predicted_trajectory = [initial_state.copy()]
        state = initial_state.copy()
        for k in range(self.horizon):
            v, w = optimal_controls[k]
            state = self.predict_state(state, v, w, self.dt)
            self.predicted_trajectory.append(state.copy())
        
        self.predicted_controls = optimal_controls
        
        # Apply first control input (receding horizon principle)
        v_opt, w_opt = optimal_controls[0]
        
        # Update previous control for next iteration
        self.prev_v = v_opt
        self.prev_w = w_opt
        
        return v_opt, w_opt
    
    def __call__(self, state, reference_trajectory, current_time, dt):
        """
        Make the controller callable for use in simulation.
        
        Args:
            state: Current state dict
            reference_trajectory: Full reference trajectory
            current_time: Current simulation time
            dt: Time step (not used, MPC has its own dt)
            
        Returns:
            v, w: Control inputs
        """
        return self.solve(state, reference_trajectory, current_time)


# Create MPC controller wrapper for the simulator API
def mpc_controller_factory(horizon=15, dt=0.1, Q=None, R=None, Q_terminal=None):
    """
    Factory function to create an MPC controller with specific parameters.
    
    Args:
        horizon: Prediction horizon
        dt: Time step
        Q: State cost matrix
        R: Control cost matrix
        Q_terminal: Terminal cost matrix
        
    Returns:
        Controller function compatible with simulator
    """
    mpc = MPCController(
        horizon=horizon,
        dt=dt,
        Q=Q,
        R=R,
        Q_terminal=Q_terminal,
        max_velocity=2.0,
        max_angular_velocity=2.0,
        max_accel=2.0
    )
    
    return mpc

# ==================== Example Controllers ====================

def feedforward_controller(state, reference_trajectory, current_time, dt):
    """Simple feedforward controller - just follows reference velocities."""
    if not reference_trajectory:
        return 0.0, 0.0
    
    # Find closest reference point by time
    ref_idx = min(range(len(reference_trajectory)), 
                  key=lambda i: abs(reference_trajectory[i][0] - current_time))
    
    _, _, _, _, ref_v, ref_w = reference_trajectory[ref_idx]
    return ref_v, ref_w


def pd_tracking_controller(state, reference_trajectory, current_time, dt, 
                           kp_v=2.0, kp_w=3.0, kd_v=0.5, kd_w=0.5):
    """
    PD controller for trajectory tracking.
    
    Args:
        kp_v, kp_w: Proportional gains for linear and angular velocity
        kd_v, kd_w: Derivative gains for linear and angular velocity
    """
    if not reference_trajectory:
        return 0.0, 0.0
    
    # Find closest reference point by time
    ref_idx = min(range(len(reference_trajectory)), 
                  key=lambda i: abs(reference_trajectory[i][0] - current_time))
    
    ref_t, ref_x, ref_y, ref_theta, ref_v, ref_w = reference_trajectory[ref_idx]
    
    # Position errors
    error_x = ref_x - state['x']
    error_y = ref_y - state['y']
    
    # Transform errors to robot frame
    cos_theta = math.cos(state['theta'])
    sin_theta = math.sin(state['theta'])
    error_forward = error_x * cos_theta + error_y * sin_theta
    error_lateral = -error_x * sin_theta + error_y * cos_theta
    
    # Orientation error
    error_theta = math.atan2(math.sin(ref_theta - state['theta']), 
                            math.cos(ref_theta - state['theta']))
    
    # Velocity errors
    error_v = ref_v - state['v']
    error_w = ref_w - state['w']
    
    # PD control law
    v = ref_v + kp_v * error_forward + kd_v * error_v
    w = ref_w + kp_w * error_theta + kd_w * error_w
    
    return v, w


def pure_pursuit_trajectory_controller(state, reference_trajectory, current_time, dt, 
                                       lookahead_time=1.0):
    """
    Pure Pursuit controller adapted for time-parameterized trajectories.
    
    Args:
        lookahead_time: How far ahead in time to look (seconds)
    """
    if not reference_trajectory:
        return 0.0, 0.0
    
    # Find lookahead point
    target_time = current_time + lookahead_time
    target_idx = min(range(len(reference_trajectory)), 
                     key=lambda i: abs(reference_trajectory[i][0] - target_time))
    
    _, target_x, target_y, _, target_v, _ = reference_trajectory[target_idx]
    
    # Calculate steering angle
    dx = target_x - state['x']
    dy = target_y - state['y']
    target_angle = math.atan2(dy, dx)
    angle_diff = math.atan2(math.sin(target_angle - state['theta']), 
                           math.cos(target_angle - state['theta']))
    
    # Control law
    v = target_v
    w = 2.5 * angle_diff
    
    return v, w


# ==================== Example Usage ====================

# ==================== Example Usage ====================

if __name__ == "__main__":
    DURATION = 15
    # Create car
    car = DifferentialDriveCar(initial_x=0, initial_y=0, initial_theta=0)
    
    # Create renderer
    renderer = SceneRenderer(figsize=(14, 11))
    
    # Generate reference trajectory (try different trajectories!)
    print("Generating reference trajectory...")
    
    # Option 1: S-curve trajectory
    reference_trajectory = TrajectoryGenerator.s_curve(
        length=15, amplitude=4, velocity=1.0, duration=DURATION, dt=0.1
    )
    
    # Option 2: Circle trajectory
    # reference_trajectory = TrajectoryGenerator.circle(
    #     center=(8, 8), radius=5, angular_velocity=0.3, duration=25.0, dt=0.1
    # )
    
    # Option 3: Figure eight trajectory
    # reference_trajectory = TrajectoryGenerator.figure_eight(
    #     center=(7, 0), radius=4, period=20.0, duration=40.0, dt=0.1
    # )
    
    # Option 4: Custom waypoints
    # waypoints = [(0, 0), (5, 2), (10, 5), (12, 8), (10, 12), (5, 10), (0, 8)]
    # velocities = [1.0, 1.5, 1.0, 0.8, 1.2, 1.0]
    # reference_trajectory = TrajectoryGenerator.custom_waypoints(
    #     waypoints, velocities, dt=0.1
    # )
    
    print(f"Generated trajectory with {len(reference_trajectory)} points")
    print(f"Duration: {reference_trajectory[-1][0]:.2f} seconds")
    
    # Create simulator
    sim = Simulator(car, renderer, dt=0.1)
    sim.set_reference_trajectory(reference_trajectory)
    
    # ==================== Choose a Controller ====================
    
    # Option 1: MPC Controller (RECOMMENDED - shows predicted trajectory!)
    print("\n=== Running MPC Controller ===")
    print("MPC Parameters:")
    print("  - Prediction Horizon: 15 steps (1.5 seconds)")
    print("  - State weights Q: [10, 10, 5] (x, y, theta)")
    print("  - Control weights R: [0.1, 0.1] (v, w)")
    print("  - Terminal weights Q_terminal: [50, 50, 20]")
    print("  - Constraints: |v| <= 2.0 m/s, |w| <= 2.0 rad/s")
    print("\nStarting simulation...\n")
    
    mpc = mpc_controller_factory(
        horizon=10,
        dt=0.1,
        Q=np.diag([10.0, 10.0, 5.0]),        # State cost: [x, y, theta]
        R=np.diag([0.1, 0.1]),               # Control cost: [v, w]
        Q_terminal=np.diag([50.0, 50.0, 20.0])  # Terminal cost
    )
    
    sim.run(
        controller=mpc,
        duration=None,  # Use trajectory duration
        render_interval=0.5,  # Render every 3 steps (MPC is slower)
        verbose=True
    )
    
    # Option 2: PD Controller
    # print("\n=== Running PD Tracking Controller ===")
    # sim.run(
    #     controller=pd_tracking_controller,
    #     duration=None,
    #     render_interval=5,
    #     verbose=True
    # )
    
    # Option 3: Feedforward controller (open-loop)
    # print("\n=== Running Feedforward Controller ===")
    # sim.run(
    #     controller=feedforward_controller,
    #     duration=None,
    #     render_interval=5,
    #     verbose=True
    # )
    
    # Option 4: Pure Pursuit
    # print("\n=== Running Pure Pursuit Controller ===")
    # sim.run(
    #     controller=pure_pursuit_trajectory_controller,
    #     duration=None,
    #     render_interval=5,
    #     verbose=True
    # )
    
    # Show final result
    print("\n" + "="*60)
    print("Simulation finished!")
    print("="*60)
    print("\nVisualization shows:")
    print("  - Blue dashed line: Reference trajectory")
    print("  - Red gradient: Actual trajectory")
    print("  - Magenta line with dots: MPC predicted trajectory (receding horizon)")
    print("  - Small magenta arrows: Predicted orientations")
    print("\nClose the plot window to exit.")
    renderer.show()