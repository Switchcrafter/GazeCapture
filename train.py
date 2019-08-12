import argparse
import json
import os
import re
import subprocess

from azureml.core import Experiment, Datastore, ScriptRunConfig, Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration, DEFAULT_GPU_IMAGE, DEFAULT_CPU_IMAGE
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

ws = Workspace.from_config()
datastore = Datastore.get(ws, 'deepeyes_dataset')
run_config = RunConfiguration(framework='Python')

default_name = az_nickname() + '_' + str(1)

vm_sizes = [ vm['name'] for vm in AmlCompute.supported_vmsizes(workspace=ws)]
vm_sizes = list(filter(lambda x: re.match('^Standard_N[CD]', x), vm_sizes))

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--compute', type=str, choices=['cluster'], default='cluster')
parser.add_argument('--vm-size', type=str, choices=vm_sizes, default='Standard_NC6')
parser.add_argument('--name', type=str, default=default_name)
parser.add_argument('--test', action='store_true')
parser.add_argument('--cluster-name', type=str)
parser.add_argument('--show-output', action='store_true', default=False)

args = parser.parse_args()

if args.compute in ['demand', 'cluster']:
    run_config.environment.docker.enabled = True
    run_config.environment.docker.gpu_support = True
    run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
    dependencies = CondaDependencies.create(conda_packages=['numpy', 'pillow', 'scipy', 'pytorch-gpu', 'torchvision'])
    run_config.environment.python.conda_dependencies = dependencies

    script_params = {
        '--data_path' : datastore.as_mount(),
        '--output_path' : './outputs'
    }

    if args.test:
        script_params['--epochs'] = 1

if args.compute == 'demand':
    run_config.target = 'amlcompute'
    run_config.amlcompute.vm_size  = args.vm_size

elif args.compute == 'cluster':

    if not args.cluster_name:
        print('cluster-name must be specified')
        exit(1)

    cluster = ComputeTarget(workspace=ws, name=args.cluster_name)
    run_config.target = cluster

elif args.compute == 'local':
    run_config.environment.python.user_managed_dependencies = True

project_dir='./pytorch'
experiment_name='gc_' + args.name

experiment = Experiment(ws, name=experiment_name)

if args.compute == 'demand':

    src = ScriptRunConfig(source_directory = project_dir, script = 'main.py', run_config = run_config)

elif args.compute == 'cluster':

    src = PyTorch(source_directory=project_dir, 
                  script_params=script_params,
                  compute_target=cluster,
                  entry_script='main.py',
                  use_gpu=True,
                  pip_packages=['numpy==1.17.0', 'Pillow==6.1.0', 'scipy==1.3.0'])

elif args.compute == 'local':
    print('local compute is TODO')
    exit(1)

if args.compute in ['cluster', 'demand']:
    run = experiment.submit(src)
    run.wait_for_completion(args.show_output)
