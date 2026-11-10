#import setup_path 
import airsim

import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("fly")
new_position_airsim = airsim.Vector3r(0,0, 0)
level_quaternion = airsim.Quaternionr(0, 0, 0, 1) # 드론 수평 유지
new_pose = airsim.Pose(new_position_airsim, level_quaternion)
client.simSetVehiclePose(new_pose,True,'Follower0')
#client.moveToPositionAsync(0, 0, -10, 5).join()

print("reset")
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(5)
print("done")

print("fly")
client.moveToPositionAsync(0, 0, -1, 5).join()
