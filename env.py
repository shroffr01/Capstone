import traci
from sumolib import checkBinary
import numpy as np
import pandas as pd

class TrafficEnv:
    def __init__(self, mode='binary'):
        # If the mode is 'gui', it renders the scenario.
        if mode == 'gui':
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.sumoCmd = [self.sumoBinary, "-c", 'C:/Users/shrof/.vscode/UVACODE/RLTC_repo_Staunton/sumo-marl/scenario/osm.sumocfg', '--no-step-log', '-W']
        
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
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            current_action = traci.trafficlight.getPhase(intersection_ID)
            if actions[i] == current_action:
                continue
            else:
                traci.trafficlight.setPhase(intersection_ID, actions[i])  # switch to next phase after yellow light

    def step(self, actions):
        self.apply_action(actions)
        for _ in range(self.decision_time):
            traci.simulationStep()
            self.time += 1

        state = self.get_state()
        reward = self.get_reward()
        done = self.get_done()
        return state, reward, done

    def get_reward(self):
        reward = [0.0 for _ in range(self.n_intersections)]
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                reward[i] += traci.lane.getLastStepHaltingNumber(lane)

        reward = -np.array(reward)
        return reward

    def get_signal_timings(self):

        #list = [list() for i in range(enumerate(traci.trafficlight.getIDList()))]
        timing_list = []
        phase_list = []
        intersections_list = traci.trafficlight.getIDList()

        for intersection_ID in (traci.trafficlight.getIDList()):

            signal_timings = traci.trafficlight.getSpentDuration(intersection_ID)
            timing_list.append(signal_timings)

            phase = traci.trafficlight.getPhase(intersection_ID)
            phase_list.append(phase)

        df = pd.DataFrame({"Intersection_ID": intersections_list, "Signal_Timings": timing_list, "Phase": phase_list})
        return df
    
    def get_done(self):
        #return traci.simulation.getMinExpectedNumber() == 0
        return self.time > 1000
    
    def close(self):
        traci.close()



if __name__ == "__main__":
    env = TrafficEnv()
    state = env.reset()
