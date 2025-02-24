import importlib

def get_model(model_name: str):

    submodule_name, class_name = model_name.split('.')

    module = importlib.import_module(f'models.{submodule_name}')

    print(module, class_name)

    return getattr(module, class_name)

if __name__ == '__main__':

    md = get_model('cnns.Resnet1x1')
    md = md(n_classes=15, n_bands=3)
    print(md)