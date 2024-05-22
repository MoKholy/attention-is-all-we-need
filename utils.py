import yaml


def load_configs():
    file = "./configs/configs.yaml"
    with open("config.yaml", "r") as file:
        configs = yaml.safe_load(file)
    return configs
