'''
API Layer
This convenience func preserves name and docstring
'''
from functools import wraps
from flask import request, jsonify
from apilayer.apis import ApiLayer


def add_method(self):
    '''
    Defining add method decorator
    '''
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)
        setattr(self, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly same as func
        return func  # returning func means func can still be used normally
    return decorator


class ExtApiLayer(ApiLayer):
    '''
    API LAYER class defination
    '''

    def __init__(self, name, port, max_hung_time, log_level, config_manager_addr=None):
        super().__init__(name=name, port=port, max_hung_time=max_hung_time, log_level=log_level,
                         config_manager_addr=config_manager_addr)

        @self.app.route("/", methods=["GET"])
        @self.app.route("/<string:text>", methods=["POST"])
        def endpoint_not_present(text=None):
            return "this is a wild endpoint\n", 404

        @self.app.route('/process', methods=["POST"])
        def temp_input():
            print("entering endpoint !!")
            data = request.get_json() #request.get_data()
            img_string = data['images']
            md = data['metadata']
            response = self.process_input(md, img_string)
            return response
