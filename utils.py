import yaml

def save_hyperparams(args, save_path, print_summary: bool = False):
    hyperparams = {}
    for key, value in vars(args).items():
        hyperparams[key] = value #local_scope[param]
    with open(save_path, 'w') as f:
        yaml.dump(hyperparams, f)
    if print_summary:
        print("Hyperparameter Summary:")
        print(open(save_path).read())
