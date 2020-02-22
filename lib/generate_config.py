import json

__all__ = ['generate_exp']

def generate_exp(experiments_dir):
    index_offset = 0
    learning_rates = [0.001,0.01,0.1]
    n_epochs = [5]
    transformation_modes = [False,True]
    flow_index = [0,1]
    integration_index = [0,1]

    id = index_offset
    for k, tm in enumerate(transformation_modes):
        for i, lr in enumerate(learning_rates):
            for j, ne in enumerate(n_epochs):
                for l, fi in enumerate(flow_index):
                    for m, ii in enumerate(integration_index):
                        if not tm and l > 0:
                            continue
                        exp = {'id':id,
                                'lr':lr,
                                'n_epochs':ne,
                                'transform_mode':tm,
                                'flow_index':fi,
                                'integration_index':ii,
                                'model_path':f'model_pt/{id}.pt'}
                        path = f'{experiments_dir}/{id}.json'
                        with open(path,'w') as f:
                            json.dump(exp, f)
                        id += 1
