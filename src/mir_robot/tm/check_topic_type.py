import time
from mir_driver.rosbridge import RosbridgeSetup

MIR_IP = "192.168.0.177"
MIR_PORT = 9090

def main():
    robot = RosbridgeSetup(MIR_IP, MIR_PORT)
    time.sleep(1)
    
    resp = robot.callService('/rosapi/topic_type', msg={"topic": "/move_base_node/global_costmap/inflated_obstacles"})
    print("Type of inflated_obstacles:", resp)
    
if __name__ == "__main__":
    main()
