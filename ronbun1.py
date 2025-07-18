from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
import random
import matplotlib.pyplot as plt

def compute_S(model):
    return sum(1 for agent in model.schedule.agents if agent.state =="S")
def compute_E(model):
    return sum(1 for agent in model.schedule.agents if agent.state =="E")
def compute_I(model):
    return sum(1 for agent in model.schedule.agents if agent.state =="I")
def compute_C(model):
    return sum(1 for agent in model.schedule.agents if agent.state =="C")
def compute_R(model):
    return sum(1 for agent in model.schedule.agents if agent.state =="R")
def compute_INT(model):
    return sum(1 for agent in model.schedule.agents if agent.state =="INT")

class SmartPhoneAgent(Agent):
    """
    スマートフォンエージェントのクラス
    """
    def __init__(self, unique_id, model, android_share):
        super().__init__(unique_id, model)
        self.state = "S"  # 状態を初期化
        self.os = "Android" if self.random.random() < android_share else "Other"
        self.latency_timer = 0
        self.direction = self.random.choice([(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)])
        self.movement_mode = None  # 移動モードを初期化
        self.movement_timer = 0  # 移動タイマーを初期化
        self.infecting_agent = None  # 感染エージェントを初期化

    def move(self):
        """
        エージェントの移動メソッド
        """
        # エージェントの移動方法を選択
        if self.movement_timer <= 0:
            self.movement_mode = self.random.choice(["RW", "SL","PAUSE"])
            # RW: ランダムウォーク, SL: ステップランダムウォーク, PAUSE: 停止
            self.movement_timer = self.random.randint(1, 5)
            if self.movement_mode =="SL":
                self.direction = self.random.choice([(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)])
        
        # RWの実装
        if self.movement_mode == "RW":
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False
            )
            empty_steps = [p for p in possible_steps if self.model.grid.is_cell_empty(p)]

            if len(empty_steps) > 0:
                new_position = self.random.choice(empty_steps)
                self.model.grid.move_agent(self, new_position)

        #SLの実装
        elif self.movement_mode == "SL":
            next_pos = (self.pos[0] + self.direction[0], self.pos[1] + self.direction[1])
            if not self.model.grid.out_of_bounds(next_pos) and self.model.grid.is_cell_empty(next_pos):
                self.model.grid.move_agent(self, next_pos)
            # else:
            #     # 移動できない場合はランダムに方向を変える
            #     self.direction = self.random.choice([(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)])
            
        # PAUSEの実装
        elif self.movement_mode == "PAUSE":
            # 停止状態では何もしない
            pass
        # タイマーをデクリメント
        self.movement_timer -= 1

    def update_state(self):
        """
        エージェントの状態を更新するメソッド
        """
        if self.state == "R":
            if self.random.random() < self.model.renewal_rate:
                self.state = "S"
            return
        
        if self.state == "INT":
            self.state = "S"
            return
        
        #回復状態に遷移する条件
        if self.state == "I":
            if self.random.random() < self.model.recover_rate:
                self.state = "R"
            return
        
        #E状態に遷移する条件
        if self.state == "E":
            is_still_valid = False
            if self.infecting_agent is not None:
                neighbor_cells = self.model.grid.get_neighborhood(self.pos, moore=True)
                neighbors = self.model.grid.get_cell_list_contents(neighbor_cells)
                if self.infecting_agent in neighbors and self.infecting_agent.state == "I":
                    is_still_valid = True
            if is_still_valid:
                self.latency_timer -= 1
                if self.latency_timer <=0:
                    if self.os == "Android": self.state = "I"
                    else: self.state = "C"
                    self.infecting_agent = None
            else:
                self.state = "INT"
                self.infecting_agent = None
            return

        #自分がS状態でない場合は何もしない
        if self.state != "S":
            return
        #周囲のエージェントを取得
        neighbor_cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbor_cells)

        #周囲にいるI状態のエージェントがいるか確認
        infected_neighbors = [agent for agent in neighbors if agent.state == "I"]
        if len(infected_neighbors) > 0:
            if self.random.random() < self.model.infection_rate:
                self.state = "E"
                self.latency_timer = self.model.latency_time
                self.infecting_agent = self.random.choice(infected_neighbors)
            

    def step(self):
        self.move()
        self.update_state()


class BlueToothWormModel(Model):
    """
    BlueToothワームの拡散モデル
    """
    def __init__(self, mum_agents=100, width=100, height=100, android_share=0.84, infection_rate=0.9,initial_infected=1, latency_time=5,recover_rate=0.05,renewal_rate=0.01):
        super().__init__()
        self.mum_agents = mum_agents
        self.android_share = android_share
        self.infection_rate = infection_rate
        self.latency_time = latency_time
        self.recover_rate = recover_rate
        self.renewal_rate = renewal_rate
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.initial_infected_positions = []

        self.datacollector = DataCollector(
            model_reporters={
                "S": compute_S,
                "E": compute_E,
                "I": compute_I,
                "C": compute_C,
                "R": compute_R,
                "INT": compute_INT
            }
        )

        for i in range(self.mum_agents):
            a = SmartPhoneAgent(i, self, self.android_share)
            self.schedule.add(a)
            pos = self.random.choice(list(self.grid.empties))
            self.grid.place_agent(a, pos)
        
        #初期感染者の設定
        infected_agents = self.random.sample(self.schedule.agents, initial_infected)
        for agent in infected_agents:
            agent.state = "I"
            self.initial_infected_positions.append(agent.pos)
    

    def step(self):
        """
        モデルのステップを実行するメソッド
        """
        self.datacollector.collect(self)
        self.schedule.step()


# モデルの実行
if __name__ == "__main__":
    simulation_steps = 800
    model = BlueToothWormModel(mum_agents=9000, width=100, height=100, android_share=0.84, infection_rate=0.9, initial_infected=1, latency_time=5, recover_rate=0.01, renewal_rate=0.01)
    print(f"初期感染者の位置: {model.initial_infected_positions}")
    for i in range(simulation_steps):
        model.step()

    results = model.datacollector.get_model_vars_dataframe()
    results.plot()
    plt.title("Bluetooth Worm Model Results")
    plt.xlabel("Step")
    plt.ylabel("mumber of Agents")
    plt.xlim(0, simulation_steps)
    plt.grid(True)
    plt.savefig('test1_plot.png')
    plt.close()