from typing import Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class PointRobot(gym.Env):
    def __init__(self, id=None, seed=None):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.hazard_size = 0.8
        self.hazard_position_list = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]
        self.goal_position = np.array([2.2, 2.2])
        self.goal_size = 0.3
        self.dt = 0.05
        self.state = None
        self.id = id
        self.seed(seed)
        self.last_dist = None
        self.steps = 0

        self._max_episode_steps = 300
        self.con_dim = 1

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.action_space.seed(seed)
        return [seed]
    
    def reset(self,state=None) -> np.ndarray:
        self.state = np.random.uniform(low=[-3.0, -3.0, 0.5, np.pi / 4], high=[3.0, 3.0, 2.0, 3 * np.pi / 4])
        if self.id is not None:
            if state is None:
                self.state = np.array([-1.8, 0.0, 2.0, np.pi/4], dtype=np.float32) # feasible no violation, near 
                # self.state = np.array([-1.5, 0.0, 2.0, np.pi/4], dtype=np.float32) # infeasible , can find safest policy
                # self.state = np.array([-2.7, -2.7, 2.0, np.pi/2], dtype=np.float32)
            else:
                self.state=state
        self.last_dist = np.linalg.norm([self.state[0]-self.goal_position[0], self.state[1]-self.goal_position[1]])
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        state = self.state + self._dynamics(self.state, action) * self.dt
        reward, done = self.reward_done(state)
        info = self.get_info(state)
        info.update({'theta':state[3]})
        self.state = state
        assert done == self.check_done(state)
        return self._get_obs(), reward, done, info

    def reward_done(self, state):
        reward = 0.0
        done = False
        dist = np.linalg.norm([state[0]-self.goal_position[0], state[1]-self.goal_position[1]])
        
        reward += (self.last_dist - dist)

        self.last_dist = dist

        if dist <= self.goal_size:
            reward += 1
            done = True

        if (abs(state[0])>3.0 or abs(state[1])>3.0):
            done = True

        return reward, done
    
    def get_info(self, state):
        min_dist = float('inf')
        for hazard_pos in self.hazard_position_list:
            hazard_vec = hazard_pos[:2] - state[:2]
            dist = np.linalg.norm(hazard_vec)
            min_dist = min(dist, min_dist)
        con_val = self.hazard_size - min_dist
        info = dict(
            cost=int(con_val<=0),
            constraint_value=con_val,
            violation=(con_val>0).item()
        )
        assert np.isclose(info['constraint_value'], self.get_constraint_values(state), atol=1e-4)
        assert info['violation'] == self.check_violation(state)
    
        return info

    def get_constraint_values(self, states):
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2
        # for batched states, compute the distances to the closest hazard
        # in a vectorized way
        min_dist = np.full(states.shape[0], float('inf'))
        for hazard_pos in self.hazard_position_list:
            hazard_vec = hazard_pos[:2] - states[:, :2]
            dist = np.linalg.norm(hazard_vec, axis=1)
            min_dist = np.minimum(dist, min_dist)
        return self.hazard_size - np.squeeze(min_dist)
    
    def check_violation(self, states):
        if len(states.shape) == 1:
                states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return self.get_constraint_values(states) > 0

    def check_done(self, states):
        if len(states.shape) == 1:
                states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        out_of_bound = np.logical_or(
            np.logical_or(states[:, 0] < -3.0, states[:, 0] > 3.0),
            np.logical_or(states[:, 1] < -3.0, states[:, 1] > 3.0)
        )

        reach_goal = (np.linalg.norm(states[:, :2] - self.goal_position, axis=1) <= self.goal_size)
        done = np.logical_or(out_of_bound, reach_goal)

        done = done.item() if len(done.shape) == 0 else done
        return done
    
    @staticmethod
    def _dynamics(s, u):
        v = s[2]
        theta = s[3]

        dot_x = v * np.cos(theta)
        dot_y = v * np.sin(theta)
        dot_v = u[0]
        dot_theta = u[1]

        dot_s = np.array([dot_x, dot_y, dot_v, dot_theta], dtype=np.float32)
        return dot_s

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)
        obs[:3] = self.state[:3]
        theta = self.state[3]
        obs[3] = c = np.cos(theta)
        obs[4] = s = np.sin(theta)
        rot_mat = np.array([[c, -s],
                            [s, c]], dtype=np.float32)

        i = 0
        for hazard_pos in self.hazard_position_list:

            x, y = (hazard_pos[:2] - self.state[:2]) @ rot_mat
            hazard_vec = x + 1j * y

            dist = np.abs(hazard_vec)
            angle = np.angle(hazard_vec)

            obs[5+i*3] = dist
            obs[6+i*3] = np.cos(angle)
            obs[7+i*3] = np.sin(angle)

            i += 1

        return obs

    def _get_avoidable(self, state):
        x, y, v, theta = state

        for hazard_position in self.hazard_position_list:
            hazard_vec = hazard_position - np.array([x, y])

            dist = np.linalg.norm(hazard_vec)
            if dist <= self.hazard_size:
                return False


            velocity_vec = np.array([v * np.cos(theta), v * np.sin(theta)])
            velocity = np.linalg.norm(velocity_vec)
            velocity = np.clip(velocity, 1e-6, None)
            cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            delta = self.hazard_size ** 2 - (dist * sin_theta) ** 2
            if cos_theta <= 0 or delta < 0:
                continue

            acc = self.action_space.low[0]
            if np.cross(velocity_vec, hazard_vec) >= 0:
                omega = self.action_space.low[1]
            else:
                omega = self.action_space.high[1]
            action = np.array([acc, omega])
            s = np.copy(state)
            while s[2] > 0:
                s = s + self._dynamics(s, action) * self.dt
                dist = np.linalg.norm([hazard_position[0]-s[0], hazard_position[1]-s[1]])
                if dist <= self.hazard_size:
                    return False
            
        return True

    def plot_map(self, ax, v: float = 2.0, theta: float = np.pi / 4):
        from matplotlib.patches import Circle

        n = 200
        xs = np.linspace(-3.0, 3.0, n)
        ys = np.linspace(-3.0, 3.0, n)
        xs, ys = np.meshgrid(xs, ys)
        vs = v * np.ones_like(xs)
        thetas = theta * np.ones_like(xs)
        obs = np.stack((xs, ys, vs, np.cos(thetas), np.sin(thetas)), axis=-1)

        avoidable = np.zeros_like(xs)
        for i in range(n):
            for j in range(n):
                avoidable[i, j] = float(self._get_avoidable([xs[i, j], ys[i, j], v, theta]))
        ax.contour(xs, ys, avoidable - 0.5, levels=[0], colors='k', linewidths=2, linestyles='--')

        for hazard_position in self.hazard_position_list:
            circle = Circle((hazard_position[0], hazard_position[1]), self.hazard_size, fill=False, color='k',linewidth=1.5)
            ax.add_patch(circle)

        ax.set_xlim([-3,3])
        ax.set_ylim([-3,3])
        return ax
        
    
    def plot_task(self, ax):
        from matplotlib.patches import Circle

        n = 200
        xs = np.linspace(-3.0, 3.0, n)
        ys = np.linspace(-3.0, 3.0, n)
        xs, ys = np.meshgrid(xs, ys)


        for hazard_position in self.hazard_position_list:
            circle = Circle((hazard_position[0], hazard_position[1]), self.hazard_size, fill=True, alpha=0.5, color=(0.30,0.52,0.74))
            ax.add_patch(circle)
        # Goal
        circle = Circle((self.goal_position[0], self.goal_position[1]), self.goal_size, fill=True, alpha=0.5, color=(0.35,0.66,0.35))
        ax.add_patch(circle)
        circle = Circle((-2.7, -2.7), 0.1, fill=True, alpha=0.5, color='r')
        ax.add_patch(circle)
        ax.set_xlim([-3,3])
        ax.set_ylim([-3,3])
        return ax

    def _get_single_avoidable(self, state):
        x, y, v, theta = state
        hazard_position = self.hazard_position_list[1]

        hazard_vec = hazard_position - np.array([x, y])

        dist = np.linalg.norm(hazard_vec)
        if dist <= self.hazard_size:
            return False


        velocity_vec = np.array([v * np.cos(theta), v * np.sin(theta)])
        velocity = np.linalg.norm(velocity_vec)
        velocity = np.clip(velocity, 1e-6, None)
        cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        delta = self.hazard_size ** 2 - (dist * sin_theta) ** 2
        if cos_theta <= 0 or delta < 0:
            return True

        acc = self.action_space.low[0]
        if np.cross(velocity_vec, hazard_vec) >= 0:
            omega = self.action_space.low[1]
        else:
            omega = self.action_space.high[1]
        action = np.array([acc, omega])
        s = np.copy(state)
        while s[2] > 0:
            s = s + self._dynamics(s, action) * self.dt
            dist = np.linalg.norm([hazard_position[0]-s[0], hazard_position[1]-s[1]])
            if dist <= self.hazard_size:
                return False
            
        return True

    def plot_single_map(self, ax, color, v: float = 2.0, theta: float = np.pi / 4):
        from matplotlib.patches import Circle

        n = 200
        xs = np.linspace(-3.0, 3.0, n)
        ys = np.linspace(-3.0, 3.0, n)
        xs, ys = np.meshgrid(xs, ys)
        vs = v * np.ones_like(xs)
        thetas = theta * np.ones_like(xs)

        avoidable = np.zeros_like(xs)
        for i in range(n):
            for j in range(n):
                avoidable[i, j] = float(self._get_single_avoidable([xs[i, j], ys[i, j], v, theta]))
        ax.contour(xs, ys, avoidable - 0.5, levels=[0], colors=color, linewidths=1, linestyles='--')

        return ax