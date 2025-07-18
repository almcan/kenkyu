import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# パラメータ    
K = 500 # ノード数
avg_degree = 4 # 平均次数
p0 = 0.1
# 感染・回復のパラメータ
alpha = 0.01    #感染率
beta = 0.006    #マルウェア除去率
delta = 0.001   #脆弱性修復率
eta = 0.0002    #マルウェアの脆弱性発見率

#　ゲーム理論のパラメータ
theta = 5.0 #ゲームの実行率
kappa = 0.1  #ペアワイズ・フェルミのパラメータ
T = 1.0     #利得計算の時間
w1, w2, w3 = 1.0, 0.1, 0.0 #重み

# シュミレーション時間
max_time = 5000
#モデル名
SCENARIO = 'volunteer'

G = nx.barabasi_albert_graph(K, avg_degree)

for i in G.nodes():
    is_countermeasure = random.random() < p0
    if is_countermeasure:
        G.nodes[i]['state'] = 'SC'
    else:
        G.nodes[i]['state'] = 'SN'
    
    G.nodes[i]['last_update_time'] = 0.0
    G.nodes[i]['time_uninfected'] = 0.0
    G.nodes[i]['time_in_group'] = T if is_countermeasure else 0.0

initial_infected_node = 0
if G.nodes[initial_infected_node]['state'] == 'SC':
    G.nodes[initial_infected_node]['state'] = 'IC'
else:
    G.nodes[initial_infected_node]['state'] = 'IN'

print("初期設定完了")

def calculate_rates(G):
    """
    ノードの状態に基づいて感染率、回復率、脆弱性修復率を計算する関数
    """
    #カウント用
    num_SN, num_SC, num_IN, num_IC, num_RN, num_RC = 0, 0, 0, 0, 0, 0
    rates = {}

    #感染イベントレート
    infection_rate = 0
    infected_nodes = [i for i, data in G.nodes(data=True) if data['state'] in ['IN', 'IC']]
    for i in infected_nodes:
        for neighbor in G.neighbors(i):
            if G.nodes[neighbor]['state'] in ['SN', 'SC']:
                infection_rate += alpha
    
    for i in G.nodes():
        state = G.nodes[i]['state']
        if state == 'SN':
            num_SN += 1
        elif state == 'SC':
            num_SC += 1
        elif state == 'IN':
            num_IN += 1
        elif state == 'IC':
            num_IC += 1
        elif state == 'RN':
            num_RN += 1
        elif state == 'RC':
            num_RC += 1
    
    num_infected = num_IN + num_IC
    num_susceptible = num_SN + num_SC

    #各イベントの合計レート
    rates['infection'] = infection_rate
    rates['elimination'] = beta * num_infected
    rates['repair'] = delta * num_susceptible
    rates['vul_discovery'] = eta * num_infected
    rates['game'] = theta

    total_rate = sum(rates.values())

    return rates, total_rate, (num_SN, num_SC, num_IN, num_IC, num_RN, num_RC)

def execute_infection_event(G):
    """
    感染イベントが起こる関数
    """
    possible_infection_links = []
    infected_nodes = [i for i, data in G.nodes(data=True) if data['state'] in ['IN', 'IC']]

    for infector in infected_nodes:
        for neighbor in G.neighbors(infector):
            if G.nodes[neighbor]['state'] in ['SN', 'SC']:
                possible_infection_links.append(neighbor)
    
    if not possible_infection_links:
        return
    
    node_to_infect = random.choice(possible_infection_links)

    if G.nodes[node_to_infect]['state'] == 'SN':
        G.nodes[node_to_infect]['state'] = 'IN'
    elif G.nodes[node_to_infect]['state'] == 'SC':
        G.nodes[node_to_infect]['state'] = 'IC'

def execute_elimination_event(G): 
    """
    感染ノードを回復状態に更新する関数
    """
    infection_nodes = [i for i, data in G.nodes(data=True) if data['state'] in ['IN', 'IC']]
    if not infection_nodes:
        return
    node_to_eliminate = random.choice(infection_nodes)
    if G.nodes[node_to_eliminate]['state'] == 'IN':
        G.nodes[node_to_eliminate]['state'] = 'RN'
    elif G.nodes[node_to_eliminate]['state'] == 'IC':
        G.nodes[node_to_eliminate]['state'] = 'RC'
        
def execute_repair_event(G):
    susceptible_nodes = [i for i, data in G.nodes(data=True) if data['state'] in ['SN', 'SC']]
    if not susceptible_nodes:
        return
    
    node_to_repair = random.choice(susceptible_nodes)
    if G.nodes[node_to_repair]['state'] == 'SN':
        G.nodes[node_to_repair]['state'] = 'RN'
    elif G.nodes[node_to_repair]['state'] == 'SC':
        G.nodes[node_to_repair]['state'] = 'RC'

def execute_vulnerability_discovery_event(G, scenario): 
    """
    マルウェアが脆弱性を発見し、回復ホストを感受性状態に戻す関数
    """
    # 1. 現在の対策グループの人数(m)をカウント
    m = 0
    for i, data in G.nodes(data=True):
        if data['state'] in ['SC', 'IC', 'RC']:
            m += 1
            
    # 2. 対策グループが先に発見していた確率を計算
    prob_group_wins = m / K
    
    # 3. 確率に基づいてどちらのケースが起こるか決定
    if random.random() < prob_group_wins:
        # --- ケースA: 対策グループが発見済みの脆弱性だった場合 ---
        
        if scenario == 'volunteer':
            # ボランティアモデル：全員が情報を知っているため、誰も状態変化しない
            pass
            
        elif scenario == 'ally':
            # 同盟モデル：グループ外(RN)のホストだけが影響を受ける
            nodes_to_update = []
            for node_id, data in G.nodes(data=True):
                if data['state'] == 'RN':
                    nodes_to_update.append((node_id, 'SN'))
            
            for node, new_state in nodes_to_update:
                G.nodes[node]['state'] = new_state
                
    else:
        # --- ケースB: マルウェアが未知の脆弱性を発見した場合 ---
        nodes_to_update = []
        for node_id, data in G.nodes(data=True):
            if data['state'] == 'RN':
                nodes_to_update.append((node_id, 'SN'))
            elif data['state'] == 'RC':
                nodes_to_update.append((node_id, 'SC'))
        
        for node, new_state in nodes_to_update:
            G.nodes[node]['state'] = new_state
        
def execute_game_event(G, current_time):
    """
    ゲーム理論に元図いてノードを更新する関数
    """
    #ペイオフの計算
    payoffs = {}
    benefits = {}

    for i in G.nodes():
        time_elapsed = current_time - G.nodes[i]['last_update_time']
        if G.nodes[i]['state'] not in ['IN','IC']:
            G.nodes[i]['time_uninfected'] += time_elapsed
        
        if G.nodes[i]['state'] in ['SN', 'IC', 'SC']:
            G.nodes[i]['time_in_group'] += time_elapsed
        
        benefit_i = (w1 * (G.nodes[i]['time_uninfected'] / T) +
                     w2 * (G.nodes[i]["time_in_group"] / T) - 
                     w3 * (G.nodes[i]['time_in_group'] / T ))
        benefits[i] = benefit_i

        G.nodes[i]['last_update_time'] = current_time
        G.nodes[i]['time_uninfected'] = 0.0
        G.nodes[i]['time_in_group'] = 0.0
    
    for i in G.nodes():
        payoff_i = 0
        for neighbor in G.neighbors(i):
            payoff_i += (benefits[i] - benefits[neighbor])
        payoffs[i] = payoff_i
    
    node_to_review = random.choice(list(G.nodes()))

    neighbors = list(G.neighbors(node_to_review))
    if not neighbors:
        return
    
    best_neighbor = -1
    max_payoff = -np.inf
    for neighbor in neighbors:
        if payoffs[neighbor] > max_payoff:
            max_payoff = payoffs[neighbor]
            best_neighbor = neighbor
    
    payoff_diff = payoffs[best_neighbor] - payoffs[node_to_review]
    update_prob = 1 / (1 + np.exp(-payoff_diff / kappa))

    if random.random() < update_prob:
        best_neighbor_is_countermeasure = G.nodes[best_neighbor]['state'] in ['SC','IC', 'RC']
        current_node_state = G.nodes[node_to_review]['state']

        if best_neighbor_is_countermeasure: # 対策グループに参加する
            if current_node_state == 'SN': G.nodes[node_to_review]['state'] = 'SC'
            elif current_node_state == 'IN': G.nodes[node_to_review]['state'] = 'IC'
            elif current_node_state == 'RN': G.nodes[node_to_review]['state'] = 'RC'
        else: # 対策グループから離脱する
            if current_node_state == 'SC': G.nodes[node_to_review]['state'] = 'SN'
            elif current_node_state == 'IC': G.nodes[node_to_review]['state'] = 'IN'
            elif current_node_state == 'RC': G.nodes[node_to_review]['state'] = 'RN'
            
current_time = 0.0
history = []

while current_time < max_time:
    #1.全イベントのレートと合計レートを計算
    rates, total_rate, counts = calculate_rates(G)
    history.append([current_time] + list(counts))

    if total_rate == 0:
        print("全てのノードが感染していないか、回復済みです。シミュレーションを終了します。")
        break

    #2.次のイベントまでの時間を決定
    time_to_next_event = np.random.exponential(1.0 / total_rate)
    current_time += time_to_next_event

    #3.イベントの選択
    event_names = list(rates.keys())
    event_probs = [r / total_rate for r in rates.values()]
    chosen_event = np.random.choice(event_names, p=event_probs)

    #4.選択されたイベントに基づいてノードの状態を更新
    if chosen_event == 'infection':
        execute_infection_event(G)
    elif chosen_event == 'elimination':
        execute_elimination_event(G)
    elif chosen_event == 'repair':
        execute_repair_event(G)
    elif chosen_event == 'vul_discovery':
        execute_vulnerability_discovery_event(G, SCENARIO)
    elif chosen_event == 'game':
        execute_game_event(G, current_time)
    
print("シミュレーション完了")

history = np.array(history)
plt.figure(figsize=(10, 6))
plt.plot(history[:,0], history[:,1] + history[:,2], label = 'Susceptible(SN + SC)')
plt.plot(history[:,0], history[:,3] + history[:,4], label='Infected(IN + IC)')
plt.plot(history[:,0], history[:,5] + history[:,6], label='Recovered(RN + RC)')
plt.plot(history[:,0], history[:,2] + history[:,4] + history[:,6], label='Countermeasure (SC + IC + RC)', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Number of Hosts')
plt.legend()
plt.grid(True)
plt.savefig("ronbun3_plot.png")