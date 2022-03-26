'''
Script that allows users to play Messenger in the terminal.
'''
import argparse
import gym
import messenger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_id',
        type=str,
        required=True,
        help='environment id for human play')
    args = parser.parse_args()

    env = gym.make(args.env_id)

    # map from keyboard entry to gym action space
    action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3, '': 4}

    keep_playing = 'yes'
    total_games = 0
    total_wins = 0

    while keep_playing.lower() not in ['no', 'n']:
        obs, manual = env.reset()
        done = False
        eps_reward = 0
        eps_steps = 0
        reward = 0
        env.render()
        action = input('\nenter action [w,a,s,d,\'\']: ')

        while not done:
            if action.lower() in action_map:
                obs, reward, done, info = env.step(action_map[action])
                eps_steps += 1
                eps_reward += reward
                env.render()

                # if reward != 0:
                #     print(f'\ngot reward: {reward}\n')
            if done:
                total_games += 1
                if reward == 1:
                    total_wins += 1
                    print('\n\tcongrats! you won!!\n')
                else:
                    print('\n\tyou lost :( better luck next time.\n')
                break

            action = input('\nenter action [w,a,s,d,\'\']: ')

        print(
            f'\nFinished episode with reward {eps_reward} in {eps_steps} steps!\n')
        keep_playing = input('play again? [n/no] to quit: ')
        # if keep_playing.lower() not in ['no', 'n']:
        #     clear_terminal()

    print(
        f'\nThanks for playing! You won {total_wins} / {total_games} games.\n')
