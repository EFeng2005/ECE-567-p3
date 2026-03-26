# Tier 1 official commands copied from:
# external/ogbench/impls/hyperparameters.sh
#
# These are the author-provided command lines for the subset we plan to reproduce.

# antmaze-large-navigate-v0 (CRL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1
# antmaze-large-navigate-v0 (HIQL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
# antmaze-large-navigate-v0 (QRL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003
# antmaze-large-navigate-v0 (GCIQL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3

# humanoidmaze-medium-navigate-v0 (CRL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-medium-navigate-v0 (HIQL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
# humanoidmaze-medium-navigate-v0 (QRL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-medium-navigate-v0 (GCIQL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995

# cube-double-play-v0 (CRL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# cube-double-play-v0 (HIQL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# cube-double-play-v0 (QRL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# cube-double-play-v0 (GCIQL)
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0

# scene-play-v0 (CRL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# scene-play-v0 (HIQL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# scene-play-v0 (QRL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# scene-play-v0 (GCIQL)
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0

# puzzle-3x3-play-v0 (CRL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-3x3-play-v0 (HIQL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# puzzle-3x3-play-v0 (QRL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-3x3-play-v0 (GCIQL)
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0

