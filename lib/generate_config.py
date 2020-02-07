import json

__all__ = ['generate_exp']

def generate_exp(experiments_dir):
    learning_rates = [0.001,0.005,0.01]
    n_epochs = [10]
    transformation_modes = [False]

    for i, lr in enumerate(learning_rates):
        for j, ne in enumerate(n_epochs):
            for k, tm in enumerate(transformation_modes):
                id = i * len(n_epochs) * len(transformation_modes) \
                    + j * len(transformation_modes) + k
                exp = {'id':id,
                        'lr':lr,
                        'n_epochs':ne,
                        'transform_mode':tm,
                        'model_path':f'model_pt/{id}.pt'}
                path = f'{experiments_dir}/{id}.json'
                with open(path,'w') as f:
                    json.dump(exp, f)
