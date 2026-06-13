import time
from mir_driver.rosbridge import RosbridgeSetup

MIR_IP = "192.168.0.177"
MIR_PORT = 9090

def main():
    robot = RosbridgeSetup(MIR_IP, MIR_PORT)
    time.sleep(1)
    if not robot.is_connected():
        print("Not connected")
        return
    
    print("Calling /rosapi/topics ...")
    try:
        resp = robot.callService('/rosapi/topics', msg={}, timeout=10.0)
        if resp and 'topics' in resp:
            for t in resp['topics']:
                if 'costmap' in t.lower() or 'map' in t.lower() or 'planner' in t.lower():
                    print(t)
        else:
            print("No topics in response:", resp)
    except Exception as e:
        print("Error calling service:", e)

if __name__ == "__main__":
    main()
