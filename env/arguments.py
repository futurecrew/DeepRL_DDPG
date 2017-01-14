import argparse

def get_game_name(rom):
    if '/' in rom:
        game = rom.split('/')[-1]
    else:
        game = rom
    if '.' in game:
        game = game.split('.')[0]
    return game

def get_args():
    parser = argparse.ArgumentParser()    
    
    parser.add_argument('bin', type=str, help='torcs bin name')
    parser.add_argument('port', type=int, help='torcs port number')
    parser.add_argument('--vision', action='store_true', help='use vision input or not')
    parser.add_argument('--track', type=int, default=-1, help='track file no')
    parser.add_argument('--epoch_step', type=int, default=10000, help='train step no per epoch')
    parser.add_argument('--backend', type=str, default='TF')
    parser.add_argument('--thread-no', type=int, default=1, help='Number of multiple threads for Asynchronous RL')
    parser.add_argument('--network', type=str, default='nips', choices=['nips', 'nature'], help='network model nature or nips') 
    parser.add_argument('--drl', type=str, default='dqn', choices=['dqn', 'double_dqn', 'prioritized_rank', 'prioritized_proportion', 'a3c_lstm', 'a3c', '1q'])
    parser.add_argument('--snapshot', type=str, default=None, help='trained file to resume training or to replay') 
    parser.add_argument('--device', type=str, default='', help='gpu or cpu')
    parser.add_argument('--show-screen', action='store_true', help='whether to show display or not')
    parser.set_defaults(vision=False)
    parser.set_defaults(show_screen=False)
    
    args = parser.parse_args()
    args.game = 'torcs'
    args.rom = 'torcs'
    
    if args.rom == 'vizdoom':
        args.env = 'vizdoom'
        from env.vizdoom_env import initialize_args
        initialize_args(args)
    elif args.rom == 'ale':
        args.env = 'ale'
        from env.ale_env import initialize_args
        initialize_args(args)
    elif args.rom == 'torcs':
        args.env = 'torcs'
        from env.torcs_env import initialize_args
        initialize_args(args)
    
    return args
