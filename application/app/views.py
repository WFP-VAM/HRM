from app import application
from flask import render_template, request
import yaml
import os
import sys
sys.path.append(os.path.join("../","Src"))
sys.path.append(os.path.join("app", "src"))

try:
    config_file = 'app_config.yml'
    with open(config_file, 'r') as cfgfile:
        config = yaml.load(cfgfile)
except FileNotFoundError:
    print('config file {} not found.'.format(config_file))


@application.route('/')
def home():
    return render_template('index.html')


# downscale endpoint
@application.route('/downscale', methods=['POST'])
def master():
    print('-> downscale of survey data.')
    from app.endpoints.downscale import downscale
    return downscale(config, request)