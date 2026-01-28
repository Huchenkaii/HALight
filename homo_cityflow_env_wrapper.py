import json
import numpy as np
import cityflow as cf
import os

class Intersection:
    def __init__(self, config, all_roads):
        self.config = config
        self.id = config["id"]
        self.point = np.array([config["point"]["x"], config["point"]["y"]])
        self.enter_roads = set()
        self.movement = {}
        self.lane_types = {}
        self.lane_roads = {}

        for roadlink in config.get("roadLinks", []):
            start_road = roadlink["startRoad"]
            end_road = roadlink["endRoad"]
            self.enter_roads.add(start_road)
            move_type = roadlink.get("type", "").lower()

            for lane_link in roadlink.get("laneLinks", []):
                in_lane = f"{start_road}_{lane_link['startLaneIndex']}"
                out_lane = f"{end_road}_{lane_link['endLaneIndex']}"

                if in_lane not in self.movement:
                    self.movement[in_lane] = []

                self.movement[in_lane].append(out_lane)
                self.lane_roads[in_lane] = start_road

                if in_lane not in self.lane_types:
                    self.lane_types[in_lane] = set()

                if "left" in move_type:
                    self.lane_types[in_lane].add("left_turn")
                elif "right" in move_type:
                    self.lane_types[in_lane].add("right_turn")
                elif "straight" in move_type:
                    self.lane_types[in_lane].add("straight")

        self.road_dir_map = self._compute_directions_by_points(all_roads)


    def _compute_directions_by_points(self, all_roads):
        dir_map = {}
        cx, cy = self.point
        for road_id in self.enter_roads:
            road_info = next((r for r in all_roads if r["id"] == road_id), None)
            if not road_info:
                continue
            pts = np.array([[p["x"], p["y"]] for p in road_info["points"]], dtype=np.float32)
            dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
            near_idx = np.argmin(dists)
            far_idx = np.argmax(dists)
            far_pt = pts[far_idx]
            dx, dy = far_pt[0] - cx, far_pt[1] - cy
            if abs(dx) > abs(dy):
                direction = "east" if dx > 0 else "west"
            else:
                direction = "north" if dy > 0 else "south"
            dir_map[road_id] = direction
        return dir_map

class HomoCityflowEnv:
    """
    cityflow homogeneous environment
    """

    def __init__(self, args):
        self.args = args
        with open(args.net, "r") as f:
            self.roadnet = json.load(f)

        # 初始化路口对象
        self.intersections = {}
        self.intersection_ids = []
        for inter in sorted(
            [i for i in self.roadnet["intersections"] if not i.get("virtual", True) and "trafficLight" in i],
            key=lambda x: int(x["id"])
        ):
            self.intersections[inter["id"]] = Intersection(inter, self.roadnet["roads"])
            self.intersection_ids.append(inter["id"])

        self.speed_threshold = 1.39
        self.vehicle_waiting_time = {}
        self.vehicle_enter_leave_dict = dict()
        self.previous_vehicles_list = {}

        # 初始化引擎
        cityflow_config = {
            "interval": 1,
            "seed": self.args.seed,
            "laneChange": False,
            "dir": args.dir,
            "roadnetFile": self.args.net,
            "flowFile": self.args.flow,
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
        self.delta_time = args.delta_time
        self.current_phases = {iid: 0 for iid in self.intersection_ids}

        self.obs_drop_prob = getattr(args, "obs_drop_prob", 0.0)
        self.obs_drop_mask_dict = {}

    # -------------------------------------------------------
    def reset(self):
        self.eng = cf.Engine(self.config_path, thread_num=4)
        self.current_phases = {iid: 0 for iid in self.intersection_ids}
        self.vehicle_enter_leave_dict = dict()
        self.previous_vehicles_list = {}
        self.vehicle_waiting_time = {}

        self.obs_drop_mask_dict = {}
        for inter_id, inter in self.intersections.items():
            num_in_lanes = len(inter.movement.keys())
            lane_mask = np.random.rand(num_in_lanes) < self.obs_drop_prob
            self.obs_drop_mask_dict[inter_id] = lane_mask

        return self._get_homo_state()

    # -------------------------------------------------------
    def step(self, actions, pressreward=False):
        for iid, act in actions.items():
            self.eng.set_tl_phase(iid, act)
            self.current_phases[iid] = act
        for _ in range(self.delta_time):
            self.eng.next_step()
            self._update_enter_leave_time()

            vehicle_speeds = self.eng.get_vehicle_speed()

            vehicles = self.eng.get_vehicles()
            for vehicle_id in vehicles:
                vehicle_speed = vehicle_speeds.get(vehicle_id, 0)

                if vehicle_speed < self.speed_threshold:
                    if vehicle_id in self.vehicle_waiting_time:
                        self.vehicle_waiting_time[vehicle_id] += 1
                    else:
                        self.vehicle_waiting_time[vehicle_id] = 1

        state = self._get_homo_state()

        if pressreward:
            reward = self._get_press_reward()
        else:
            reward = self._get_reward()

        done = self.eng.get_vehicle_count() == 0
        return state, reward, done, {}

    def _apply_lane_disturbance(self, inter_id, lane_vehicle_count, lane_waiting_vehicle_count):
        inter = self.intersections[inter_id]
        lane_mask = self.obs_drop_mask_dict.get(inter_id, np.zeros(len(inter.movement), dtype=bool))

        disturbed_vehicle_count = dict(lane_vehicle_count)
        disturbed_waiting_vehicle_count = dict(lane_waiting_vehicle_count)

        for idx, in_lane in enumerate(inter.movement.keys()):
            if lane_mask[idx]:
                disturbed_vehicle_count[in_lane] = 0
                disturbed_waiting_vehicle_count[in_lane] = 0

        return disturbed_vehicle_count, disturbed_waiting_vehicle_count

    def _get_homo_state(self):
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        homo_state = {}

        for iid, inter in self.intersections.items():
            disturbed_vehicle_count, disturbed_waiting_vehicle_count = self._apply_lane_disturbance(
                iid, lane_vehicle_count, lane_waiting_vehicle_count
            )

            dir_state = {
                d: {"left_turn": [0, 0], "straight": [0, 0], "right_turn": [0, 0]}
                for d in ["north", "east", "south", "west"]
            }

            for in_lane in inter.movement.keys():
                road_id = inter.lane_roads[in_lane]
                lane_type_set = inter.lane_types.get(in_lane, {"unknown"})
                direction = inter.road_dir_map.get(road_id, None)
                if not direction or "unknown" in lane_type_set:
                    continue

                wave = disturbed_vehicle_count.get(in_lane, 0)
                queue = disturbed_waiting_vehicle_count.get(in_lane, 0)

                weight = 1.0 / len(lane_type_set)
                for lane_type in lane_type_set:
                    dir_state[direction][lane_type][0] += wave * weight
                    dir_state[direction][lane_type][1] += queue * weight

            vec = []
            for d in ["north", "east", "south", "west"]:
                for f in ["left_turn", "straight", "right_turn"]:
                    wave, queue = dir_state[d][f]
                    vec.extend([wave, queue])
            vec.append(self.current_phases[iid])
            homo_state[iid] = np.array(vec, dtype=np.float32)

        return homo_state

    def _get_reward(self):
        lane_wait = self.eng.get_lane_waiting_vehicle_count()
        reward = {}
        for iid, inter in self.intersections.items():
            total_wait = sum(lane_wait.get(l, 0) for l in inter.movement.keys())
            reward[iid] = -total_wait
        return reward

    def _get_press_reward(self):
        reward = {}

        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        max_capacity = 1.0

        for iid, inter in self.intersections.items():
            total_pressure = 0.0

            for in_lane, out_lanes in inter.movement.items():
                incoming_cnt = lane_vehicle_count.get(in_lane, 0.0)

                if not isinstance(out_lanes, (list, tuple)):
                    out_lanes = [out_lanes]

                for out_lane in out_lanes:
                    outgoing_cnt = lane_vehicle_count.get(out_lane, 0.0)

                    pressure = (incoming_cnt / max_capacity) - (outgoing_cnt / max_capacity)
                    total_pressure += abs(pressure)

            reward[iid] = -total_pressure

        return reward

    def _update_enter_leave_time(self):
        current_vehicles_list = self.eng.get_vehicles(include_waiting=True)
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

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()

    def get_throughput(self):
        finish_trip_count = 0
        for info in self.vehicle_enter_leave_dict.values():
            if not info['leave_time'] is None:
                finish_trip_count += 1
        return finish_trip_count

    def get_average_queue_time(self):
        total_queue_time = 0.0
        finished_vehicle_count = 0

        for vid, info in self.vehicle_enter_leave_dict.items():
            if info["leave_time"] is not None:
                if vid in self.vehicle_waiting_time:
                    total_queue_time += self.vehicle_waiting_time[vid]
                    finished_vehicle_count += 1

        if finished_vehicle_count > 0:
            return total_queue_time / finished_vehicle_count
        else:
            return 0.0

    def get_intersections(self):
        return self.intersection_ids
