import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from marl.env import TrafficEnv
from marl.maddpg import MADDPG
from marl.utils import get_average_travel_time


from pathlib import Path

def project_root():
    # two levels up: marl/ → Capstone/
    return Path(__file__).resolve().parent.parent

def abspath_under_project(*parts):
    return str(project_root().joinpath(*parts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-R", "--render", action="store_true",
        help= "whether render while training or not"
    )
    parser.add_argument(
        "--sumo-cfg", default=None,
        help="Path to your .sumocfg file (overrides default under /scenario)"
    )
    parser.add_argument(
        "--results-dir", default=abspath_under_project("results"),
        help="Where to write trained models & performance plots"
    )    
    args = parser.parse_args()

    # Before the start, should check SUMO_HOME is in your environment variables
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # create directories
    os.makedirs(args.results_dir, exist_ok=True)

    # config paths
    sumo_cfg = args.sumo_cfg or abspath_under_project("scenario", "osm.sumocfg")
    model_path = os.path.join(args.results_dir, "trained_model.th")
    plot_path  = os.path.join(args.results_dir, "performance.png")
    csv_path   = os.path.join(args.results_dir, "traffic_signals.csv")

    # # configuration
    # state_dim = 38 # need to change this number to the maximum padded amount, can also modify to certain number
    # action_dim = 2
    # n_agents = 16
    # # intersections is 3!, need to make this a dynamic thing, temporary solution: count # of intersections in network
    # n_episode = 5

    # configuration
    action_dim = 2
    n_episodes = 5

    env = TrafficEnv(sumo_cfg, gui=args.render)
    first_state = env.reset()
    env.close()

    n_agents = first_state.shape[0]
    state_dim = first_state.shape[1]

    print(f"Detected {n_agents} intersections, state_dim={state_dim}")

    agent = MADDPG(n_agents, state_dim, action_dim)
    env = TrafficEnv(sumo_cfg, gui=args.render)

    # Create an Environment and RL Agent
    # env = TrafficEnv(sumo_cfg, gui=args.render)
    # agent = MADDPG(n_agents, state_dim, action_dim)

    # Train your RL agent
    performance_list = []
    df_original = pd.DataFrame() # intialize dataframe to store results

    for episode in range(n_episodes):
        state = env.reset()
        reward_epi = []
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False

        while not done:
            # select action according to a given state
            for i in range(n_agents):
                action, action_prob = agent.select_action(state[i, :], i)
                actions[i] = action
                action_probs[i] = action_prob

            # apply action and get next state and reward
            before_state = state
            state, reward, done, df = env.step(actions)

            # make a transition and save to replay memory
            transition = [before_state, action_probs, state, reward, done]
            agent.push(transition)

            # train an agent
            if agent.train_start():
                for i in range(n_agents):
                    agent.train_model(i)
                agent.update_eps()

            if done:
                break

            df_original = pd.concat([df_original, df], ignore_index=True) # appending df to original

        env.close()
        average_traveling_time = get_average_travel_time()
        performance_list.append(average_traveling_time)

        print(f"Episode: {episode+1}/{n_episodes}    Average Traveling Time:{average_traveling_time:.2f}   Eps:{agent.eps:.3f}")

    # Save the model
    agent.save_model(model_path)

    # Save the dataframe
    df_original.to_csv(csv_path, index=False)
    
    # Plot the performance
    plt.style.use('ggplot')
    plt.figure(figsize=(10.8, 7.2), dpi=120)
    plt.plot(performance_list)
    plt.xlabel('# of Episodes')
    plt.ylabel('Average Traveling Time')
    plt.title('Performance of MADDPG for controlling traffic lights')
    plt.savefig(plot_path)



  

