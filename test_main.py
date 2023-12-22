from testers import ddqn_eval, ppo_eval, a3c_eval

def main():
    """
    Main function for orchestrating testing different models.
    """
    # All DDQN Agents
    ddqn_eval.test()
    # PPO Agent
    ppo_eval.test()
    # A3C Agent
    a3c_eval.test()

if __name__ == '__main__':
    main()