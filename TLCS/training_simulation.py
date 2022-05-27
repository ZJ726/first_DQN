import traci
import numpy as np
import random
import timeit
import os
# sumo配置
# 对应environment.net.xml的相位码
PHASE_NS_GREEN = 0  # action 0 code 00，对应action_number
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        """
        运行仿真的一个回合， 然后开始训练session
        """
        start_time = timeit.default_timer()   #计时器

        # 首先, 产生仿真的路网文件 并 设置sumo
        self._TrafficGen.generate_routefile(seed=episode)  #随机产生每一个episode
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0  #变体
        # 基本: current_total_wait=0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # 得到交叉路口当前状态
            current_state = self._get_state()

            # 基本奖励函数：当前行动的累计奖励(动作之间的全部等待时间)
            #new_total_wait = self._collect_total_waiting_times()
            # reward = 0.9 * old_total_wait - new_total_wait

            # 变体奖励函数：当前行动的累计奖励 (动作之间的累计等待时间)
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # 将数据保存在memory中
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))  #四元组组成的一个样本

            # 根据当前交叉口状态，选择一个信号灯相位激活
            action = self._choose_action(current_state, epsilon)

            # 若当前选择相位（动作）不同于上一相位（动作），激活黄灯
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # 继续之前，执行相位
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # 保存之后和累计的奖励的变量
            old_state = current_state
            old_action = action
            # old_total_wait = new_total_wait    基本奖励函数
            old_total_wait = current_total_wait  #变体奖励函数

            # 只保存有意义的奖励
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()  #调用数据统计和可视化
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))  #四舍五入两位小数
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1) #已用时间-开始时间

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):   #循环所有epoch
            self._replay()  #开始训练
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        当产生统计数据时，执行sumo的step
        """
        if (self._step + steps_todo) >= self._max_steps:  # 不要做超过最大允许的步骤数
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # 启动step 1
            self._step += 1 # 更新step数
            steps_todo -= 1
            queue_length = self._get_queue_length
            self._sum_queue_length += queue_length  #排队数加一
            self._sum_waiting_time += queue_length # 1 step = 1 s，所以 排队长度=等待秒数

    # 基本奖励函数
    # def _collect_total_waiting_times(self):
    #     """
    #     检索来向车道每辆车的等待时间
    #     """
    #     incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    #     car_list = traci.vehicle.getIDList()
    #     for car_id in car_list:
    #         wait_time = traci.vehicle.getWaitingTime(car_id)
    #         road_id = traci.vehicle.getRoadID(car_id)  # 得到车辆所在车道的id
    #         if road_id in incoming_roads:  # 只考虑来向车道的等待时间
    #             self._waiting_times[car_id] = wait_time
    #         else:
    #             if car_id in self._waiting_times:  # 一辆被跟踪的汽车已经通过了十字路口
    #                 del self._waiting_times[car_id]
    #     total_waiting_time = sum(self._waiting_times.values())
    #     return total_waiting_time

    # 变体奖励函数
    def _collect_waiting_times(self):
        """
        检索来向车道每辆车的等待时间
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]   #即驶向信号灯的车道
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)  #通过car_id得到车辆的等待时间
            road_id = traci.vehicle.getRoadID(car_id)  # 得到车辆所在车道的id
            if road_id in incoming_roads:  # 若只考虑来向车道的等待时间，则将wait_time加到_waiting_times字典中
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # 一辆被跟踪的汽车已经通过了十字路口
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())   #以列表形式返回字典所有值
        return total_waiting_time


    def _choose_action(self, state, epsilon):
        """
        根据epsilon-greedy策略，决定是否采取探索或利用行为
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # 随机选择，即探索
        else:
            return np.argmax(self._Model.predict_one(state)) # 根据state，按贪心策略选择最大的q值，即利用


    def _set_yellow_phase(self, old_action):
        """
        激活黄灯
        """
        yellow_phase_code = old_action * 2 + 1 # 基于上一动作，获得黄色相位码
        traci.trafficlight.setPhase("TL", yellow_phase_code)  #traci函数：把信号灯设置为指定相位


    def _set_green_phase(self, action_number):
        """
        在sumo中激活恰当的绿灯相位组合
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    @property
    def _get_queue_length(self):
        """
        检索每条来向车道上速度=0的车辆数
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        """
        从sumo中检索交叉路口的状态， 以cell占用的形式
        """
        state = np.zeros(self._num_states)  #创建新的全0状态数组
        car_list = traci.vehicle.getIDList()   #获得车辆id，返回列表

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)   #得到车辆在车道上的位置
            lane_id = traci.vehicle.getLaneID(car_id)   #得到车辆所在的车道id
            lane_pos = 750 - lane_pos  # 倒置车道位置，若车辆接近信号灯 -> lane_pos = 0

            # 距离交通灯的米数 -> 映射为cells
            if lane_pos < 7:       #车在车道上的位置距信号灯小于7m，则表示第0块cell
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # 找到车辆所在的车道
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":  #直直右
                lane_group = 0
            elif lane_id == "W2TL_3":  #左转
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # 由两个位置ID组成，创建一个区间0-79的数字
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell      #车辆在第一cell
                valid_car = True
            else:
                valid_car = False  # 没有发现过十字路口的车辆或驶离十字路口的车辆

            if valid_car:
                state[car_position] = 1  # 在状态数组中，以cell占用的形式，写下某个车辆（id）在某个位置（car_position）

        return state

    #训练过程
    def _replay(self):
        """
        在memory中检索一组样本（大小为batch_size），并为每个样本更新q公式
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # memory已满
            states = np.array([val[0] for val in batch])  # 从batch中提取状态
            next_states = np.array([val[3] for val in batch])  # 从batch中提取下一状态

            #预测
            q_s_a = self._Model.predict_batch(states)  # 为每一个样本预测 Q(state)
            q_s_a_d = self._Model.predict_batch(next_states)  # 为每一个样本预测 Q(next_state)

            # 设置训练数组，zeros((行，列))
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):  #将batch的每一个元素列举出来
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # 提取一个样本的数据
                current_q = q_s_a[i]  # 得到之前预测的 Q(state)
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # 更新 Q(state, action)
                x[i] = state
                y[i] = current_q  # 包括动作值的Q(state)

            self._Model.train_batch(x, y)  # 用更新的Q值训练神经网络


    def _save_episode_stats(self):
        """
        保存数据，可视化
        """
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # 一个episode中，每一步排队车辆的平均数

    # 私有属性在外部可读
    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

