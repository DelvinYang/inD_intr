from typing import List

import torch
from loguru import logger
from tqdm import tqdm

from common import State
from read_data_inD import DataReaderInD
from scenarioind import ScenarioInD, Vehicle

logger.info("start main")

prefix_number, data_path = '00', '/Users/delvin/Desktop/programs/跨文化返修/inD'
past_frames_needed = 12

data_reader = DataReaderInD(prefix_number, data_path)
scenario = ScenarioInD(data_reader)


all_sequence_samples = []

for ego_id in tqdm(data_reader.id_list, desc="Processing ego_ids"):
    ego_veh: Vehicle = scenario.find_vehicle_by_id(ego_id)

    start_frame = ego_veh.initial_frame
    end_frame = ego_veh.final_frame

    total_available = end_frame - start_frame

    if total_available < past_frames_needed + 1:
        continue  # 不足 12+1 帧，跳过该车

    # === 顺序采样：采 12 帧，跳过 12 帧
    step = past_frames_needed + 12
    available_start_indices = list(range(start_frame, end_frame - past_frames_needed, step))

    # logger.info(f"ego_id {ego_id}: usable sequence start frames: {available_start_indices}")

    for idx in available_start_indices:
        frame_feature_list = []
        valid = True

        for offset in range(past_frames_needed):
            frame_num = idx + offset
            try:
                ego_state: State = scenario.find_vehicle_state(frame_num, ego_id)
                svs_state: List[State] = scenario.find_svs_state(frame_num, ego_id)

                # === ego 特征 ===
                ego_feature = torch.tensor(
                    ego_state.lon + ego_state.lat,
                    dtype=torch.float
                )  # [4]

                # === 周围车特征 ===
                ego_pos = torch.tensor([ego_state.x[0], ego_state.y[0]], dtype=torch.float)
                svs_with_dist = []
                for sv in svs_state:
                    sv_pos = torch.tensor([sv.x[0], sv.y[0]], dtype=torch.float)
                    dist = torch.norm(sv_pos - ego_pos, p=2)
                    sv_feat = torch.tensor(sv.lon + sv.lat, dtype=torch.float)  # [4]
                    svs_with_dist.append((dist.item(), sv_feat))

                # 排序并截断为前 8 个
                svs_with_dist.sort(key=lambda x: x[0])
                sv_features = [feat for _, feat in svs_with_dist[:8]]

                # 不足 8 个补零
                while len(sv_features) < 8:
                    sv_features.append(torch.zeros(4))

                frame_feature = torch.stack([ego_feature] + sv_features, dim=0)  # [9, 4]
                frame_feature_list.append(frame_feature)

            except Exception as e:
                logger.warning(f"skip sample at frame {frame_num} for ego {ego_id} due to {e}")
                valid = False
                break

        if valid:
            try:
                target_state = scenario.find_vehicle_state(idx + past_frames_needed, ego_id)
                label = torch.tensor([target_state.x[2], target_state.y[2]], dtype=torch.float)

                sequence_tensor = torch.stack(frame_feature_list, dim=0)  # [T, 9, 4]
                all_sequence_samples.append((sequence_tensor, label))
            except Exception as e:
                logger.warning(f"skip label for ego {ego_id} due to {e}")
                continue

logger.info(f"Total samples collected: {len(all_sequence_samples)}")
save_path = "./sequence_tensor_dataset.pt"
torch.save(all_sequence_samples, save_path)

logger.info(f"Saved dataset with {len(all_sequence_samples)} samples to {save_path}")