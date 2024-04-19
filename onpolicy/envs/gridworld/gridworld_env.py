import copy
import numpy as np
import pkg_resources
import json
from gym import spaces


NOOP = 4
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 0

WALL = 1
EMPTY = 0
AGENT = 2
DOOR = 3
SWITCH = 4
TARGET = 5

OPENED_DOOR = 6

class WorldObj:
    repr_dict = {
        OPENED_DOOR: 'd',
        DOOR: 'D',
        EMPTY: '.',
        AGENT: '@',
        SWITCH: '>',
        WALL: '#',
        TARGET: 'G'
    }
    def __init__(self, x, y, typeobj, canpassby=False, target=None, open_on_hold=True):
        """
        x, y: coordinate of the object
        typeobj: type of the object
        canpassby: can the agents go through this object
        target: the linked object to this object, for example, agents' goals or 
            switch cells of doors
        open_on_hold: only used for doors, requires one agent to stand still on the
            switch cell's door to open it.
        """
        self.type = typeobj
        self.x = x 
        self.y = y
        self.canpassby = canpassby
        self.target = np.array(target)
        self.open_on_hold = open_on_hold 
        self.open=False

    def __getitem__(self, i):
        return getattr(self.kwargs, i, None)
    
    def __str__(self):
        tp = self.type 
        if self.type == DOOR:
            if self.open: tp = OPENED_DOOR
        return WorldObj.repr_dict[tp]

    def __repr__(self):
        return str(self)



class GridworldEnv:
    """
    :param plan: the number of scenario to read from json file
    :param separated_reward: whether the return rewards are different for each agent or to averaging them all
    """

    def __init__(
            self,
            plan: int=1,
            seed: int=None,
            separated_rewards:bool=False
    ):
        self.plan = plan
        self.max_step = 200
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]

        self.action_pos_dict = {NOOP: np.array([0, 0]), UP: np.array([-1, 0]), 
                                DOWN: np.array([1, 0]), LEFT: np.array([0, -1]), 
                                RIGHT: np.array([0, 1])}
        self.img_shape = [800, 800, 3]  # visualize state

        # initialize system state
        self.grid_map_path = pkg_resources.resource_filename(__name__, 'plan{}.json'.format(plan))

        self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map

        self.current_agent_list = copy.deepcopy(self.agent_list)
        self.start_agent_list = copy.deepcopy(self.agent_list)

        self.grid_map_shape = self.start_grid_map.shape
        self.n_agents = len(self.agent_list)
        self.n_actions = len(self.action_pos_dict)

        self.separated_rewards = separated_rewards

        self.action_space = spaces.Tuple([spaces.Discrete(5) for _ in range(self.n_agents)] )

        # seeding
        self.seed(seed)

        # counter for time step
        self.time = 0
        self.collected_goal = 0
        self._cur_step = 0

        self.goal_reward = 10
        self.collision_penalty = 0.1

        self.setup_grid_agent()
        self.current_goal_on_map = self.n_goals

        
        # each agent observation space contains the range of 3x3 block squares surrounding current agent standing
        # and the coordinate of the current agents
        self.observation_shape = 11

        self.observation_space = spaces.Tuple([spaces.Box(np.zeros(self.observation_shape, dtype=float),
                                                          np.ones(self.observation_shape, dtype=float))
                                               for _ in range(len(self.agent_list))
                                               ])
        self.state_dim = np.prod(self.grid_map_shape) * 3
        self.share_observation_space = spaces.Tuple([spaces.Box(np.zeros(self.state_dim, dtype=float),
                                                          np.ones(self.state_dim, dtype=float))
                                               for _ in range(len(self.agent_list))
                                               ])

    def setup_grid_agent(self):
        self.agent_grid = np.zeros(self.grid_map_shape, dtype=bool)
        for agent in self.current_agent_list:
            self.agent_grid[agent.x][agent.y] = 1

    def get_state(self):
        state = np.zeros((*self.grid_map_shape, 3), dtype=int)
        # the first layer is the layout of the map (only wall and door)
        # the second layer is the agent positions
        # the third layer include passable objects, include goals, opened door status
        for i in range(self.grid_map_shape[0]):
            for j in range(self.grid_map_shape[1]):
                current_cell = self.get(i, j)
                if current_cell.type == WALL or current_cell.type == DOOR:
                    state[i, j, 0] = current_cell.type
                else: state[i, j, 2] = current_cell.type
                if current_cell.type == DOOR and current_cell.open:
                    state[i, j, 2] = OPENED_DOOR
        state[..., 1] = self.agent_grid
        return state.reshape(-1)


    def get_obs(self, agent):
        obs = np.zeros(self.observation_shape, dtype=float)
        indx = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_i = i + agent.x
                new_j = j + agent.y
                if new_i >= 0 and new_i < self.grid_map_shape[0] and \
                   new_j >= 0 and new_j < self.grid_map_shape[1]:
                    current_cell = self.get(new_i, new_j)
                    obs[indx] = current_cell.type
                    if obs[indx] == DOOR and current_cell.open:
                        obs[indx] = OPENED_DOOR
                    # observe other AGENT
                    if (i != 0 or j != 0) and self.agent_grid[new_i][new_j]:
                        obs[indx] = AGENT
                else:
                    # pad outside cells with walls
                    obs[indx] = WALL 
                indx += 1
        # normalize
        obs = obs / OPENED_DOOR  # opened door has the highest value
        # corrdinate of the agents
        obs[-2] = agent.x / (self.grid_map_shape[0]-1)
        obs[-1] = agent.y / (self.grid_map_shape[1]-1)
        return obs

    def step(self, actions):
        rewards = np.zeros((self.n_agents, 1), dtype=float)
        actions = np.array(actions)
        assert np.prod(actions.shape) == self.n_agents, f"Action of the shape {actions.shape} is not valid"
        self._cur_step += 1

        # here are the list of changes to the grid map
        # these changes are made by agents, for example by open a door, 
        # or take the food (go to goal locaiton)
        # however, these changes only take effect in the next step.
        # these change do not include the change in the agent locations
        # i.e. if two agents step to the same cell
        # then one will succeed and one will fail, the order is determined by 
        # agent id, agents with smaller id take actions first
        list_of_change_queue = []

        for i, (agent, act) in enumerate(zip(self.current_agent_list, actions)):
            act = int(act)
            can_move_to = True
            current_coordinate = np.array([agent.x, agent.y], dtype=int)
            next_step = current_coordinate + self.action_pos_dict[act]

            # check if go outside of the map
            if next_step[0] < 0 or next_step[0] >= self.grid_map_shape[0] or \
               next_step[1] < 0 or next_step[1] >= self.grid_map_shape[1]:
                rewards[i] -= self.collision_penalty
                continue

            current_cell = self.get(agent.x, agent.y)     
            next_cell = self.get(*next_step)

            if next_cell.type == DOOR and next_cell.open:
                can_move_to = True
                # if the door is not open, the agent will still be penalized at the next `if` check
            elif not next_cell.canpassby: 
                # collide to wall, doors
                can_move_to = False
                rewards[i] -= self.collision_penalty

            for j, other_agent in enumerate(self.current_agent_list):
                if i == j: continue
                if next_step[0] == other_agent.x and next_step[1] == other_agent.y:
                    # collide with other agent:
                    can_move_to = False
                    rewards[i] -= self.collision_penalty

            if (next_step != current_coordinate).any() and can_move_to and current_cell.type == SWITCH:
                door_cell = self.get(*current_cell.target)
                door_cell = copy.deepcopy(door_cell)
                if door_cell.open_on_hold:
                    # agent leaves the swith cell
                    door_cell.open = False 
                    list_of_change_queue.append(
                        (
                            door_cell.x, 
                            door_cell.y,
                            door_cell
                        )
                    )
            if can_move_to:
                agent.x = next_step[0]
                agent.y = next_step[1]

            current_cell = self.get(agent.x, agent.y)
            if current_cell.type == SWITCH:
                door_cell = copy.deepcopy(self.get(*current_cell.target))
                door_cell.open = True
                list_of_change_queue.append(
                    (
                        door_cell.x, 
                        door_cell.y, 
                        door_cell
                    )
                )

            if current_cell.type == TARGET:
                agent_target = agent.target
                if agent_target is None: continue
                if not (agent_target == np.array([current_cell.x, current_cell.y], dtype=int)).all():
                    continue
                rewards[i] += self.goal_reward
                # TARAGET disappear after the agent takes it
                list_of_change_queue.append(
                    (
                        *agent_target,
                        WorldObj(*agent.target, EMPTY, canpassby=True)
                    )
                )
                self.current_goal_on_map -= 1
                agent.target = None


        # update the changes to take effect in the actual grid map
        for change in list_of_change_queue:
            self.current_grid_map[change[0], change[1]] = change[2]
        
        self.setup_grid_agent()
        obses = []
        for agent in self.current_agent_list:
            obses.append(self.get_obs(agent))
        state = self.get_state()
        # avail_actions = self.get_avail_actions()
        avail_actions=None

        # done = self.current_goal_on_map == 0 or self._cur_step >= self.max_step
        done = self._cur_step >= self.max_step
        bad_transition = self._cur_step >= self.max_step

        infos = [{} for _ in range(self.n_agents)]
        # for i in range(self.n_agents):
        #     infos[i] = {
        #         "bad_transition": bad_transition,
        #     }
        if not self.separated_rewards:
            rewards = np.sum(rewards)
            rewards = np.array([rewards for _ in range(self.n_agents)])
            rewards = rewards.reshape(-1, 1)
        return obses, [state]*self.n_agents, rewards, [done]*self.n_agents, infos, avail_actions


    def get_avail_actions(self):
        ret = []
        for i in range(len(self.agent_list)):
            ret.append(self.get_avail_agent_actions(i))
        return ret

    def get(self, i, j):
        return self.current_grid_map[i, j]

    def get_avail_agent_actions(self, i):
        """
        this function reward a list of available actions for the agent `i`
        note that the agent can still take unavailable actions, but it is likely 
        that these actions will be penalized (e.g. hit walls)
        not all cases are checked, for example hitting other agents, so agents
        can still collide if they take only available actions
        """
        avail = np.ones(self.n_actions, dtype=bool)
        agent = self.current_agent_list[i]
        agent_coord = np.array([agent.x, agent.y], dtype=int)

        for a in [LEFT, RIGHT, UP, DOWN]:
            next_step = agent_coord + self.action_pos_dict[a]
            if next_step[0] < 0 or next_step[0] >= self.grid_map_shape[0] or \
               next_step[1] < 0 or next_step[1] >= self.grid_map_shape[1]:
                avail[a] = False
                continue
            next_cell = self.get(*next_step)
            if next_cell.type == WALL:
                avail[a] = 0
            elif next_cell.type == DOOR and not next_cell.open:
                avail[a] = False
        return avail


    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map

        self.current_agent_list = copy.deepcopy(self.start_agent_list)
        self.current_goal_on_map = self.n_goals
        self.setup_grid_agent()
        self._cur_step = 0
        obses = []
        for agent in self.current_agent_list:
            obses.append(self.get_obs(agent))
        state = self.get_state()
        # avail_actions = self.get_avail_actions()
        avail_actions = None
        return obses, [state]*self.n_agents, avail_actions
    
    def set(self, i, j, v: WorldObj): 
        self.grid_map[i, j] = v

    def _read_grid_map(self, grid_map_path):
        self.map_info = _read_json_file(grid_map_path)
        self.grid_map = np.empty(self.map_info["map_size"], dtype='O')
        self.agent_list = []
        self.n_goals = 0

        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                self.grid_map[i, j] = WorldObj(i, j, EMPTY, canpassby=True)

        for wall in self.map_info["walls"]:
            self.set(*wall, WorldObj(*wall, WALL))
        
        for agent in self.map_info["agents"]:
            # self.set(agent["start_xy"], WorldObj(AGENT, target=agent["target"]))
            if agent["target"]:
                # by default, agent can pass by goals
                self.set(*agent["target"], WorldObj(*agent["target"], TARGET, canpassby=True))
                self.n_goals += 1
            self.agent_list.append(WorldObj(*agent["start_xy"], AGENT, target=agent["target"]))
        
        for door in self.map_info["doors"]:
            self.set(*door["position"], WorldObj(*door["position"], DOOR))
            self.set(*door["switch"], WorldObj(*door["switch"], SWITCH, target=door["position"], 
                            canpassby=True, open_on_hold=door["open_on_hold"]))
        return copy.deepcopy(self.grid_map)

    def _gridmap_to_image(self, render_agent=True, img_shape=None):
        # Return image from the gridmap
        pass

    def render(self):
        img = self._gridmap_to_image()
        return img
    
    def __repr__(self):
        ret = []
        grid = copy.deepcopy(self.current_grid_map)
        for agent in self.current_agent_list:
            grid[agent.x][agent.y] = agent
        for row in grid:
            ret.append("".join(str(i) for i in row))

        return "\n".join(ret)


def _read_json_file(filepath):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON format in file: {filepath}")


if __name__ == "__main__":
    env = GridworldEnv(1)
    print(env)
    print(env.grid_map)
    for i in env.current_agent_list:
        print(i.x, i.y, i.type)

    print(env.reset())


    actions = [
        [RIGHT, DOWN],
        [RIGHT, RIGHT],
        [NOOP, RIGHT],
        [NOOP, RIGHT],
        [NOOP, UP],
        [NOOP, UP],
        [RIGHT, NOOP],
        [RIGHT, DOWN],
        [RIGHT, DOWN],
        [DOWN, DOWN],
        [RIGHT, LEFT],
        [UP, LEFT],
        [RIGHT, LEFT],
        [LEFT, NOOP],
        [UP, NOOP], 
        [LEFT, NOOP],
        [LEFT, NOOP]
    ]
    for action in actions:
        all = env.step(action)
        print(all)
        state1, state2 = all[0]
        state1 = state1[:9].reshape(3, 3)
        state2 = state2[:9].reshape(3, 3)
        print("state1")
        print(state1)
        print("state2")
        print( state2)
        # env.step(action)
        print('='*10)
        print(env)
        print('='*10)
    
    print('='*10)
    print(env.reset())
    print(env)