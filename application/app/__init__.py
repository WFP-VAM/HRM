from flask import Flask

application = Flask(__name__, instance_relative_config=True)

from app import views
