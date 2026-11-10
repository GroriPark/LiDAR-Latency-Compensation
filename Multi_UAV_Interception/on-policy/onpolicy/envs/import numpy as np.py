import numpy as np
import open3d as o3d

'''
포인트 클라우드 => 지면과 물체 분리 
'''



def _get_lidar_obs(self, agent, lidar_name="LidarSensor1"):
    lidar_names = {"LidarSensor1", "LidarSensor2"}
    all_points = []

    for name in lidar_names:
        lidar_data = self.client.getLidarData(vehicle_name=agent, lidar_name=name)
        if len(lidar_data.point_cloud) < 3:
            continue

        # 1. point_cloud -> numpy (N,3) 변환
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        all_points.append(points)

    if not all_points:
        return None

    # 모든 LiDAR 합치기
    points = np.vstack(all_points)

    # 2. Open3D point cloud 객체로 변환
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 3. RANSAC으로 ground plane 추출
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05,  # 허용 오차 (단위: m)
        ransac_n=3,              # 샘플로 뽑는 점 개수
        num_iterations=100       # 반복 횟수
    )

    [a, b, c, d] = plane_model
    # inliers = 지면에 속하는 점들의 index
    ground_cloud = pcd.select_by_index(inliers)
    obstacle_cloud = pcd.select_by_index(inliers, invert=True)

    # 4. 강화학습용 관측으로 obstacle_cloud만 사용
    obstacle_points = np.asarray(obstacle_cloud.points)

    return obstacle_points 