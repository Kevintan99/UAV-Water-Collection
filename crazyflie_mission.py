#!/usr/bin/env python3
"""
Auto-generated Crazyflie trajectory from simulation
Generated at: 2025-08-27 14:51:32

Note: This trajectory accounts for drone vs bucket positioning.
      Some waypoints target the drone position directly,
      while others target where the bucket should be.
"""

import time
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

# Crazyflie URI - update this for your drone
URI = 'radio://0/80/2M/E7E7E7E7E7'
PENDULUM_LENGTH = 0.25  # meters

def fly_trajectory(scf):
    """Execute the simulated trajectory"""
    with MotionCommander(scf, default_height=0.8) as mc:
        print("Taking off...")
        time.sleep(2)
        
        # Waypoint 1: Above Water (drone target)
        # Drone flies to position
        print("Flying to Above Water: (0.50, 0.50, 0.80)")
        mc.move_distance(0.50, 0.50, 0.00, velocity=0.3)
        time.sleep(4.5)
        
        # Waypoint 2: Water Target (bucket target)
        # Drone positions to place bucket at target
        print("Flying to Water Target: (0.50, 0.50, 0.25)")
        mc.move_distance(0, 0, -0.55, velocity=0.2)
        time.sleep(4.5)
        
        # Waypoint 3: Above Water (Return) (drone target)
        # Drone flies to position
        print("Flying to Above Water (Return): (0.50, 0.50, 0.80)")
        mc.move_distance(0, 0, 0.55, velocity=0.2)
        time.sleep(6.0)
        
        # Waypoint 4: Lift Target (bucket target)
        # Drone positions to place bucket at target
        print("Flying to Lift Target: (0.80, 0.30, 0.30)")
        mc.move_distance(0.30, -0.20, -0.50, velocity=0.3)
        time.sleep(6.0)
        
        # Land
        print("Landing...")
        mc.land()
        
        print("Mission completed!")

def main():
    """Main function to connect and fly"""
    # Initialize drivers
    cflib.crtp.init_drivers()
    
    print(f"Connecting to Crazyflie at {URI}...")
    
    with SyncCrazyflie(URI, cf=None) as scf:
        print("Connected! Starting trajectory...")
        fly_trajectory(scf)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nFlight interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
