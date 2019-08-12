import os

import azureml.core
from azureml.core import Workspace, Datastore

storage_key = os.environ['DEEPEYES_STORAGE_KEY']

ws = Workspace.from_config()
datastore = Datastore.register_azure_blob_container(ws,
                                                    datastore_name='deepeyes_dataset',
                                                    container_name='gc-data-prepped',
                                                    account_name='deepeyes',
                                                    account_key=storage_key)
print('setup finished')
