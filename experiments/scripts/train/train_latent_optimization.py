from pathlib import Path
import subprocess
import sys
from nclaw.constants import SHAPE_ENVS, ENVS, SEEDS, EPOCHS, PYTHON_PATH
from nclaw.utils import get_root, get_script_parser, dict_to_hydra

def main():
    root = get_root(__file__)
    python_path = 'python' if PYTHON_PATH is None else PYTHON_PATH

    base_args, unknown = get_script_parser().parse_known_args()
    base_args = vars(base_args)
    base_cmds = [python_path, root / 'infer_material.py']

    mode = 'train'
    quality = 'low'

    for env in ENVS:
        env = 'water'
        if env == 'jelly':
            elasticity = 'stress_latent_conditioned2'
            plasticity = 'fproj_latent_conditioned2'
        elif env == 'plasticine' or env == 'water':
            elasticity = 'stress_latent_conditioned'
            plasticity = 'fproj_latent_conditioned'

        name = Path(env) / mode / f'{elasticity}-{plasticity}'

        args = base_args | {
            'env': env,
            'env/blob/material/elasticity': elasticity,
            'env/blob/material/plasticity': plasticity,
            'env.blob.material.elasticity.requires_grad': False,
            'env.blob.material.plasticity.requires_grad': False,
            'sim': quality,
            'name': name,
        }

        cmds = base_cmds + dict_to_hydra(args)
        subprocess.run([str(cmd) for cmd in cmds], shell=False)

if __name__ == '__main__':
    main()
