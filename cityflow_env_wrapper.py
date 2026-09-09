import cityflow as cf
import json
import numpy as np
import os



class Intersection:
    def __init__(self, config):
        self.config = config
        self.id = config['id']
        self.enter_roads = set()
        self.leave_roads = set()
        self.movement = {}  # key: 唯一in_lane_id, value: 唯一out_lane_id
        self.enter_lanes = {}  # key: leave_road_id, value: list of 唯一in_lane_id

        # 遍历每个roadlink（对应一种运动类型，如左转、直行）
        for roadlink in config.get("roadLinks", []):
            start_road = roadlink["startRoad"]
            end_road = roadlink["endRoad"]
            self.enter_roads.add(start_road)
            self.leave_roads.add(end_road)

            # 关键：遍历该roadlink下的所有laneLink，提取真实车道索引startLaneIndex
            for lane_link in roadlink.get("laneLinks", []):
                # 从laneLink中获取真实的"起始车道索引"（如0、1、2，对应同方向的不同车道）
                start_lane_idx = lane_link["startLaneIndex"]
                # 从laneLink中获取真实的"结束车道索引"
                end_lane_idx = lane_link["endLaneIndex"]

                # 生成唯一的车道ID：roadID_真实车道索引（与Cityflow的车道ID格式完全一致）
                in_lane = f"{start_road}_{start_lane_idx}"
                out_lane = f"{end_road}_{end_lane_idx}"

                # 避免重复添加（同一车道可能在laneLinks中重复出现）
                if in_lane not in self.movement:
                    self.movement[in_lane] = out_lane

                # 构建enter_lanes映射（同一leave_road对应多个in_lane）
                if end_road not in self.enter_lanes:
                    self.enter_lanes[end_road] = []
                if in_lane not in self.enter_lanes[end_road]:
                    self.enter_lanes[end_road].append(in_lane)

        self.enter_roads = list(self.enter_roads)
        self.leave_roads = list(self.leave_roads)
        # 打印真实车道数，验证异构差异（如A路口10个车道、B路口5个车道）
        # print(self.movement)



class CityflowEnvWrapper:
    """
    cityflow environment demo
    phase cycle: 1->3->2->4 with all stop phase 0 between consecutive phase
    """

    def __init__(self, args,flow,env_seed=0):
        self.args = args
        np.random.seed(env_seed)
        self.delta_time = args.delta_time
        self.obs_drop_prob = args.obs_drop_prob
        self.obs_drop_mask_dict = {}  # 每个路口固定的缺失掩码
        self.lane_sum = 0

        self.speed_threshold = 1.39  # 判断排队的速度阈值
        self.vehicle_waiting_time = {}  # 存储每辆车的排队时间字典
        self.flow = flow


        cityflow_config = {
            "interval": 1,
            "seed": env_seed,
            "laneChange": False,
            "dir": args.dir,
            "roadnetFile": self.args.net,
            "flowFile": self.flow,
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile": "./replay/roadnetLogFile.json",
            "replayLogFile": "./replay/replayLogFile.txt"
        }
        print(cityflow_config)
        print("obs_drop_prob", args.obs_drop_prob)
        print("=========================")

        self.config_path = os.path.join(self.args.PATH_TO_WORK_DIRECTORY, "cityflow.config")
        with open(self.config_path, "w") as json_file:
            json.dump(cityflow_config, json_file)

        # Load roadnet.json from disk
        with open(self.args.net, "r") as f:
            self.roadnet = json.load(f)

        # Initialize intersections (only those with real traffic lights)
        self.intersections = {}
        self.intersection_ids = []
        filtered_sorted_intersections = sorted(
            [inter for inter in self.roadnet["intersections"] if
             not inter.get("virtual", True) and "trafficLight" in inter],
            key=lambda x: int(x["id"])
        )

        for intersection in filtered_sorted_intersections:
            self.intersections[intersection["id"]] = Intersection(intersection)
            self.intersection_ids.append(intersection["id"])

        self.last_waiting_count = [0 for _ in range(len(self.intersection_ids))]
        self.current_phases = {key: 0 for key in self.intersection_ids}
        self.action_type = "CHOOSE PHASE"
        self.vehicle_enter_leave_dict = dict()
        self.previous_vehicles_list = {}
        self.interid2statedim = {}
        self.all_lanes = 0
        for inter_id in self.intersection_ids:
            intersection = self.intersections[inter_id]
            num_in_lanes = len(intersection.movement.keys())
            self.all_lanes += num_in_lanes
            state_dim = num_in_lanes * 2 + 1  # queue + wave + phase
            self.interid2statedim[inter_id] = state_dim


        self.interid2actiondim = {}
        for inter_id in self.intersection_ids:
            traffic_light = self.intersections[inter_id].config.get("trafficLight", {})
            lightphases = traffic_light.get("lightphases", [])
            self.interid2actiondim[inter_id] = len(lightphases)

        for inter in self.intersections.values():
            self.lane_sum += len(inter.movement.keys())
        # print("self.lane_sum", self.lane_sum)

    def step(self, actions):
        additional_log = {"pressure": {key: 0 for key in self.intersection_ids},
                          "queuelen": {key: 0 for key in self.intersection_ids}}
        done = False
        # assign actions
        for itsx, action in actions.items():
            if self.action_type == "SWITCH" or self.action_type == "twoPhaseAllPass":
                if action == 1:
                    current_phase = self.current_phases[itsx]
                    self.eng.set_tl_phase(itsx, self._next_phase(current_phase))
                    self.current_phases[itsx] = self._next_phase(current_phase)
            elif self.action_type == "CHOOSE PHASE":
                self.eng.set_tl_phase(itsx, action)
                self.current_phases[itsx] = action
            else:
                raise Exception("Unknown action type")

        for _ in range(self.delta_time):
            self.eng.next_step()  # 执行一个模拟步骤
            self._update_enter_leave_time()  # 更新车辆的进入和离开时间

            # 获取所有车辆的速度（一次性获取所有车辆的速度）
            vehicle_speeds = self.eng.get_vehicle_speed()

            # 更新每辆车的排队时间
            vehicles = self.eng.get_vehicles()  # 获取当前所有的车辆ID
            for vehicle_id in vehicles:
                # 获取车辆的速度（单位：km/h）从预先存储的字典中获取
                vehicle_speed = vehicle_speeds.get(vehicle_id, 0)

                # 如果车辆的速度小于阈值，认为它在排队
                if vehicle_speed < self.speed_threshold:
                    # 更新排队时间字典
                    if vehicle_id in self.vehicle_waiting_time:
                        self.vehicle_waiting_time[vehicle_id] += 1  # 每步累加1秒的排队时间
                    else:
                        self.vehicle_waiting_time[vehicle_id] = 1  # 如果是第一次记录该车，初始化排队时间为1秒

        state = self._get_state()
        reward = self._get_reward()
        # print(sum(reward.values()))
        if self.eng.get_vehicle_count() == 0:
            done = True
        return state, reward, done, additional_log

    def _apply_obs_disturbance(self, obs: np.ndarray, inter_id=None) -> np.ndarray:
        obs_dim = len(obs)
        if obs_dim <= 1:
            return obs

        if inter_id is not None and inter_id in self.obs_drop_mask_dict:
            mask = self.obs_drop_mask_dict[inter_id]
            obs[:-1][mask] = 0.0

        return obs

    def _get_state(self):
        state = {}
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()

        for inter_id in self.intersection_ids:
            temp_state = self._collect_waiting_queue(inter_id, lane_waiting_vehicle_count)
            temp_wave = self._collect_wave(inter_id, lane_vehicle_count)
            temp_state.extend(temp_wave)
            temp_state.append(self.current_phases[inter_id])
            temp_state = np.array(temp_state, dtype=np.float32)

            temp_state = self._apply_obs_disturbance(temp_state, inter_id)

            state[inter_id] = temp_state
        return state


    def _collect_waiting_queue(self, intersection_id, lane_waiting_vehicle_count):
        intersection = self.intersections[intersection_id]
        waiting_queue = []
        for in_lane_id in intersection.movement.keys():
            count = lane_waiting_vehicle_count.get(in_lane_id, 0)
            waiting_queue.append(count)
        return waiting_queue

    def _collect_wave(self, intersection_id, lane_vehicle_count):
        intersection = self.intersections[intersection_id]
        wave_count = []
        for in_lane in intersection.movement.keys():
            wave_count.append(lane_vehicle_count.get(in_lane, 0))
        return wave_count

    def _get_reward(self):
        reward = {}
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        # print(len(set(lane_waiting_vehicle_count.keys())))
        for id in self.intersection_ids:
            reward[id] = self._get_queue_length(id, lane_waiting_vehicle_count)
        return reward

    def _get_queue_length(self, id, lane_waiting_vehicle_count):
        if not isinstance(id, list):
            intersection_ids = [id]
        else:
            intersection_ids = id
        current_reward = 0
        for intersection_id in intersection_ids:
            intersection = self.intersections[intersection_id]
            for in_lane in intersection.movement.keys():
                current_reward += lane_waiting_vehicle_count.get(in_lane, 0)
        return -current_reward

    def _update_enter_leave_time(self):
        current_vehicles_list = self.eng.get_vehicles(include_waiting=True)
        # print(len(current_vehicles_list))
        enter_vehicles = set(current_vehicles_list) - set(self.previous_vehicles_list)
        current_time_step = self.eng.get_current_time()
        if len(enter_vehicles) > 0:
            for v in enter_vehicles:
                self.vehicle_enter_leave_dict[v] = {"enter_time": current_time_step, "leave_time": None}
        leave_vehicles = set(self.previous_vehicles_list) - set(current_vehicles_list)
        if len(leave_vehicles) > 0:
            for v in leave_vehicles:
                self.vehicle_enter_leave_dict[v]["leave_time"] = current_time_step
        self.previous_vehicles_list = current_vehicles_list

    def _next_phase(self, current_phase):
        current_phase = int(current_phase)
        if self.action_type == "twoPhaseAllPass":
            if current_phase == 9:
                return 10
            elif current_phase == 10:
                return 9
            else:
                raise Exception('wrong phase id')
        else:
            if current_phase == 1:
                return 2
            elif current_phase == 3:
                return 4
            elif current_phase == 2:
                return 3
            elif current_phase == 4:
                return 1
            else:
                print(current_phase)
                raise Exception('wrong phase id')

    def reset(self):
        self.eng = cf.Engine(self.config_path, thread_num=4)
        self.current_phases = {iid: 0 for iid in self.intersection_ids}
        self.vehicle_enter_leave_dict = dict()
        self.previous_vehicles_list = {}
        self.vehicle_waiting_time = {}  # 存储每辆车的排队时间字典


        self.obs_drop_mask_dict = {}
        for inter_id in self.intersection_ids:
            intersection = self.intersections[inter_id]
            num_in_lanes = len(intersection.movement.keys())  # 车道数量

            # 核心修正：为每个车道生成1个mask（是否缺失），长度=num_in_lanes
            lane_mask = np.random.rand(num_in_lanes) < self.obs_drop_prob  # 每个车道：True=缺失，False=不缺失

            # 将车道mask重复一次：前num_in_lanes对应queue，后num_in_lanes对应wave（保证同一车道的queue和wave同时缺失）
            # 总长度=num_in_lanes*2（对应state_dim-1，因为phase不缺失）
            mask = np.concatenate([lane_mask, lane_mask])  # 例如：lane_mask=[T,F] → mask=[T,F,T,F]

            # 存储该路口的mask（此时mask已保证queue和wave的车道级缺失一致）
            self.obs_drop_mask_dict[inter_id] = mask
            # print(inter_id,self.obs_drop_mask_dict[inter_id])

        init_states = {}
        for inter_id in self.intersection_ids:
            intersection = self.intersections[inter_id]
            num_in_lanes = len(intersection.movement.keys())
            state_dim = num_in_lanes * 2 + 1
            init_states[inter_id] = np.zeros(state_dim, dtype=np.float32)
        return init_states

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()

    def get_throughput(self):
        finish_trip_count = 0
        for info in self.vehicle_enter_leave_dict.values():
            if not info['leave_time'] is None:
                finish_trip_count += 1
        return finish_trip_count

    def get_intersections(self):
        return self.intersection_ids

    def get_average_queue_time(self):
        """
        计算已离开路网的车辆的平均排队时间（单位：秒）。
        只统计 leave_time 不为 None 的车辆。
        """
        total_queue_time = 0.0
        finished_vehicle_count = 0

        for vid, info in self.vehicle_enter_leave_dict.items():
            if info["leave_time"] is not None:
                # 只统计已离开的车辆
                if vid in self.vehicle_waiting_time:
                    total_queue_time += self.vehicle_waiting_time[vid]
                    finished_vehicle_count += 1

        if finished_vehicle_count > 0:
            return total_queue_time / finished_vehicle_count
        else:
            return 0.0


