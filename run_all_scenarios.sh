for SCENARIO in "minimal_defense" "random_defense" "maximal_attack"
do
    echo "Running $SCENARIO Scenario"
    echo "...for QHD"
    python run_qhd.py $SCENARIO

    echo "...for DQN"
    python run_dqn.py $SCENARIO

    echo "...for Tablular Q"
    python run_tabular_q.py $SCENARIO

    echo "...for REINFORCE"
    python run_reinforce.py $SCENARIO

    echo "...for PPO"
    python run_ppo.py $SCENARIO
done

