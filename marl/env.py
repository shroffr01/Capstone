import os
import traci
from sumolib import checkBinary
import numpy as np
import pandas as pd

class TrafficEnv:
    def __init__(self, sumo_cfg_path, gui=False):
        # If the mode is 'gui', it renders the scenario.
        bin_name = 'sumo-gui' if gui else 'sumo'
        self.sumoBinary = checkBinary(bin_name)
        self.sumoCmd = [self.sumoBinary, "-c", sumo_cfg_path, '--no-step-log', '-W']
        
        self.time = None
        self.decision_time = 10
        #self.n_intersections = 3 
        # problem: our simulation will not have only 3 intersections, it can have n intersections
        # solution: look at reset function 
        #self.n_phase = 2 
        # problem: each traffic light may have different # of phases ex: some allow turn on red, others don't etc
        # solution: look at reset function
        self.intersection_phases = {}

    def reset(self):
        traci.start(self.sumoCmd)
        self.n_intersections = len(traci.trafficlight.getIDList()) ### ADDED ###
        traci.simulationStep()
        self.time = 0

        ###################################### ADDED ############################
        for intersection_ID in traci.trafficlight.getIDList():
            program_logic = traci.trafficlight.getAllProgramLogics(intersection_ID)
            if program_logic:
                # Get the number of phases from the first program logic
                self.intersection_phases[intersection_ID] = len(program_logic[0].phases)
            else:
                # Default to 2 phases if no program logic is found
                self.intersection_phases[intersection_ID] = 2
        ######################################## ADDED ####################################
        return self.get_state()
    
    def get_state(self):
        # Collect all observations first
        state = []
        max_len = 0  

        temp_states = []
        for intersection_ID in traci.trafficlight.getIDList():
            observation = []
            for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                observation.append(traci.lane.getLastStepVehicleNumber(lane))
                observation.append(traci.lane.getLastStepHaltingNumber(lane))

            n_phase = self.intersection_phases[intersection_ID] ### ADDED ###
            phase = [0 for _ in range(n_phase)]
            phase[traci.trafficlight.getPhase(intersection_ID)] = 1
            observation = np.array(observation + phase)

            temp_states.append(observation)
            max_len = max(max_len, len(observation))

        ############### ADDED: Ensure uniform shape by padding ################################
        state = np.array([np.pad(obs, (0, max_len - len(obs)), mode='constant', constant_values=0) for obs in temp_states])
        #print(state)
        ################ ADDED ###############################################
        
        return state

    def apply_action(self, actions):

        timing_list = []
        phase_list = []
        intersections_list = traci.trafficlight.getIDList()
    
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            current_action = traci.trafficlight.getPhase(intersection_ID)
            
            signal_time_sofar = traci.trafficlight.getSpentDuration(intersection_ID)

            if actions[i] == current_action or signal_time_sofar <= 6:
                # ensures phase is not changed prematurely (ie. 2 second yellow light)

                signal_timings = 0
                timing_list.append(signal_timings)

                phase = traci.trafficlight.getPhase(intersection_ID)
                phase_list.append(phase)

                continue
            else:
                signal_timings = traci.trafficlight.getSpentDuration(intersection_ID)
                phase = traci.trafficlight.getPhase(intersection_ID)

                traci.trafficlight.setPhase(intersection_ID, actions[i])  

                timing_list.append(signal_timings)
                phase_list.append(phase)
                
        df = pd.DataFrame({"Intersection_ID": intersections_list, "Signal_Timings": timing_list, "Phase": phase_list})

        return df

    def step(self, actions):
        df = self.apply_action(actions)
        for _ in range(self.decision_time):
            traci.simulationStep()
            self.time += 1

        state = self.get_state()
        reward = self.get_reward()
        done = self.get_done()
        
        return state, reward, done, df

    def get_reward(self):
        reward = [0.0 for _ in range(self.n_intersections)]
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                reward[i] += traci.lane.getLastStepHaltingNumber(lane)

        reward = -np.array(reward)
        return reward
    
    def get_done(self):
        #return traci.simulation.getMinExpectedNumber() == 0
        return self.time > 1000
    
    def close(self):
        traci.close()



if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "scenario/osm.sumocfg"
    env = TrafficEnv()
    state = env.reset()
    env.close()
