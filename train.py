import argparse
import json
import subprocess

from azureml.core import Experiment, Datastore, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration, DEFAULT_GPU_IMAGE
from azureml.train.dnn import PyTorch


def az_nickname():
    args = [
        'az',
        'ad',
        'signed-in-user',
        'show'
    ]
    completed = subprocess.run(args, check=True, stdout=subprocess.PIPE)
    output = json.loads(completed.stdout.decode('ascii'))
    return output['mailNickname']


parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--name', type=str)
parser.add_argument('--test', action='store_true')
parser.add_argument('--cluster-name', type=str, required=True)
parser.add_argument('--show-output', action='store_true', default=False)
parser.add_argument('--dataset-size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--reset', action='store_true')
parser.add_argument('--sink', action='store_true')

args = parser.parse_args()

if not args.cluster_name:
    print('cluster-name must be specified')
    exit(1)

if args.name:
    name = args.name
else:
    name = az_nickname() + '_' + str(1)

ws = Workspace.from_config()
datastore = Datastore.get(ws, 'deepeyes_dataset')
run_config = RunConfiguration(framework='Python')

run_config.environment.docker.enabled = True
run_config.environment.docker.gpu_support = True
run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
dependencies = CondaDependencies.create(conda_packages=['numpy',
                                                        'pillow',
                                                        'scipy',
                                                        'pytorch-gpu',
                                                        'torchvision'])
run_config.environment.python.conda_dependencies = dependencies

script_params = {
    '--data_path': datastore.as_mount(),
    '--output_path': './outputs'
}

if args.test:
    script_params['--epochs'] = 1

if args.epochs:
    script_params['--epochs'] = args.epochs

if args.dataset_size:
    script_params['--dataset-size'] = args.dataset_size

if args.reset:
    script_params['--reset'] = ''

if args.sink:
    script_params['--sink'] = ''

cluster = ComputeTarget(workspace=ws, name=args.cluster_name)
run_config.target = cluster

project_dir = './pytorch'
experiment_name = 'gc_' + name

experiment = Experiment(ws, name=experiment_name)

src = PyTorch(source_directory=project_dir,
              script_params=script_params,
              compute_target=cluster,
              entry_script='main.py',
              use_gpu=True,
              shm_size='8g',
              pip_packages=['numpy==1.17.0', 'Pillow==6.1.0', 'scipy==1.3.0'])

run = experiment.submit(src)
run.wait_for_completion(args.show_output)
