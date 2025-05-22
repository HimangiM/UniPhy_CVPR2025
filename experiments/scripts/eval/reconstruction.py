from pathlib import Path
import subprocess
import os
from argparse import ArgumentParser
import sys
from nclaw.constants import SHAPE_ENVS, ENVS, SEEDS, EPOCHS, RENDER, PYTHON_PATH
from nclaw.utils import get_root, get_script_parser, dict_to_hydra, clean_state, diff_mse
from nclaw.ffmpeg import cat_videos

def main():
    root = get_root(__file__)
    python_path = 'python' if PYTHON_PATH is None else PYTHON_PATH

    base_args, unknown = get_script_parser().parse_known_args()
    base_args = vars(base_args)

    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = str(base_args['gpu'])

    parser = ArgumentParser()
    parser.add_argument('--skip', dest='skip', action='store_true')
    parser.add_argument('--gt', dest='gt', action='store_true')
    parser.add_argument('--render', dest='render', nargs='+', choices=['pv'], default=['pv'])
    parser.add_argument('--video', dest='video', nargs='+', choices=['pv'], default=None)
    extra_args, _ = parser.parse_known_args(unknown)
    if extra_args.video is None:
        extra_args.video = extra_args.render

    gt_mode = 'dataset'
    mode = 'train'

    quality = 'low'

    env = 'jelly'
    # env = 'plasticine'
    # env = 'water'
    # env = 'sand'
    if env == 'jelly':
        gt_args_list = []
        subargs = base_args | {
            'env': env,
            'render': RENDER,
            'sim': quality,
        }

        args = subargs | {
            'name': Path(env) / gt_mode
        }
        gt_args_list.append(args)

        if extra_args.gt:
            for args in gt_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'eval.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

        for elasticity, plasticity in [
                ['stress_latent_conditioned2', 'fproj_latent_conditioned2'],
                ]:

            epoch_args_list = []

            name = Path(env) / mode / f'{elasticity}-{plasticity}'

            exp_name = name / gt_mode / f'{300:04d}'

            args = subargs | {
                'env/blob/material/elasticity': elasticity,
                'env/blob/material/plasticity': plasticity,
                'name': exp_name,
            }

            epoch_args_list.append(args)

            for args in epoch_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'eval_latent.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

                clean_state(root / 'log' / args['name'] / 'state')

            for args in epoch_args_list:
                info = diff_mse(root / 'log' / args['name'], root / 'log' / gt_args_list[0]['name'], skip_frame=5)
                print('{}: {}'.format(args['name'], info['mse']))

    elif env == 'plasticine':
        gt_args_list = []
        subargs = base_args | {
            'env': env,
            'render': RENDER,
            'sim': quality,
        }

        args = subargs | {
            'name': Path(env) / gt_mode
        }
        gt_args_list.append(args)

        if extra_args.gt:
            for args in gt_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'eval.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

        for elasticity, plasticity in [
                ['stress_latent_conditioned3', 'fproj_latent_conditioned3'],
                ]:

            epoch_args_list = []

            name = Path(env) / mode / f'{elasticity}-{plasticity}'

            exp_name = name / gt_mode / f'{300:04d}'

            args = subargs | {
                'env/blob/material/elasticity': elasticity,
                'env/blob/material/plasticity': plasticity,
                'name': exp_name,
            }

            epoch_args_list.append(args)

            for args in epoch_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'eval_latent.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

                clean_state(root / 'log' / args['name'] / 'state')

            for args in epoch_args_list:
                info = diff_mse(root / 'log' / args['name'], root / 'log' / gt_args_list[0]['name'], skip_frame=5)
                print('{}: {}'.format(args['name'], info['mse']))

    elif env == 'water':
        gt_args_list = []
        subargs = base_args | {
            'env': env,
            'render': RENDER,
            'sim': quality,
        }

        args = subargs | {
            'name': Path(env) / gt_mode
        }
        gt_args_list.append(args)

        if extra_args.gt:
            for args in gt_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'eval.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

        for elasticity, plasticity in [
                ['stress_latent_conditioned4', 'fproj_latent_conditioned4']
                ]:

            epoch_args_list = []

            name = Path(env) / mode / f'{elasticity}-{plasticity}'

            exp_name = name / gt_mode / f'{300:04d}'

            args = subargs | {
                'env/blob/material/elasticity': elasticity,
                'env/blob/material/plasticity': plasticity,
                'name': exp_name,
            }

            epoch_args_list.append(args)

            for args in epoch_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'scripts/eval_latent.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

                clean_state(root / 'log' / args['name'] / 'state')

            for args in epoch_args_list:
                info = diff_mse(root / 'log' / args['name'], root / 'log' / gt_args_list[0]['name'], skip_frame=5)
                print('{}: {}'.format(args['name'], info['mse']))

    elif env == 'sand':
        gt_args_list = []
        subargs = base_args | {
            'env': env,
            'render': RENDER,
            'sim': quality,
        }

        args = subargs | {
            'name': Path(env) / gt_mode
        }
        gt_args_list.append(args)

        if extra_args.gt:
            for args in gt_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'eval.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

        for elasticity, plasticity in [
                ['stress_latent_conditioned5', 'fproj_latent_conditioned5']  # water
                ]:

            epoch_args_list = []

            name = Path(env) / mode / f'{elasticity}-{plasticity}'

            exp_name = name / gt_mode / f'{300:04d}'

            args = subargs | {
                'env/blob/material/elasticity': elasticity,
                'env/blob/material/plasticity': plasticity,
                'name': exp_name,
            }

            epoch_args_list.append(args)

            for args in epoch_args_list:

                if not extra_args.skip:
                    base_cmds = [python_path, root / 'scripts/eval_latent.py']
                    cmds = base_cmds + dict_to_hydra(args)
                    subprocess.run([str(cmd) for cmd in cmds], shell=False)

                clean_state(root / 'log' / args['name'] / 'state')

            for args in epoch_args_list:
                info = diff_mse(root / 'log' / args['name'], root / 'log' / gt_args_list[0]['name'], skip_frame=5)
                print('{}: {}'.format(args['name'], info['mse']))


if __name__ == '__main__':
    main()
