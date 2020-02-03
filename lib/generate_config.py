import json

__all__ = ['generate_exp']

def generate_exp(experiments_dir):
    learning_rates = [0.001,0.005]
    n_epochs = [10]

    for i, lr in enumerate(learning_rates):
        for j, ne in enumerate(n_epochs):
            id = i * len(n_epochs) + j
            exp = {'id':id,
                    'lr':lr,
                    'n_epochs':ne,
                    'model_path':f'model_pt/{id}.pt'}
            path = f'{experiments_dir}/{id}.json'
            with open(path,'w') as f:
                json.dump(exp, f)
