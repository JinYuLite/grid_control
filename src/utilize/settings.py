import yaml

def check_gen_type(gen_type):
    return all([ele == 1 or ele == 5 or ele == 2 for ele in gen_type])

# get dict attribute with 'obj.attr' format
class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

with open('utilize/parameters/main.yml', 'r') as f:
    dict_ = yaml.load(f, Loader=yaml.Loader)
    name_index = {}
    for key, val in zip(dict_["un_nameindex_key"], dict_["un_nameindex_value"]):
        name_index[key] = val
    dict_['name_index'] = name_index

settings = dotdict(dict_)
del dict_

if not check_gen_type(settings.gen_type):
    raise NotImplemented
