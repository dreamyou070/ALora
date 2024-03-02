import os
import json
config_dir = 'vae_config.json'
with open(config_dir, 'r') as f :
    jdict = json.load(f)
print(jdict)