#!/usr/bin/env python3

import os
import json
import flask
import importlib.util
import logging
import traceback
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.WARN)




def not_found_on_error(handler):
    def new_handler(*args, **kwargs):
        try:
            res, status = handler(*args, **kwargs)
        except:
            e_repr = traceback.format_exc()
            logger.error(e_repr)
            res = {
                'state': 'UNAVAILABLE',
                'status': {'error_code': 'UNKNOWN', 'error_message': e_repr},
            }
            status = 404
        return flask.Response(
            response=json.dumps(res),
            status=status,
            mimetype='application/json'
        )
    new_handler.__name__ = handler.__name__
    return new_handler


class ScoringService(object):
    models = {}

    @classmethod
    def get_model(cls, model: str):
        def get_fs_models():
            return os.listdir('/opt/models/')

        if model not in cls.models:
            assert model in get_fs_models(), f'model not found: {model}'
            spec = importlib.util.spec_from_file_location(
                f'{model}',
                f'/opt/models/{model}/{model}.py'
            )

            cls.models[model] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.models[model])

        return cls.models[model]

    @classmethod
    def summarize(cls, model, input):
        clf = cls.get_model(model)
        return clf.Model.summarize(input)

    @classmethod
    def metadata(cls, model):
        clf = cls.get_model(model)
        return clf.Model.metadata()


app = flask.Flask(__name__)


@app.route('/v1/models/<model>', methods=['GET'])
@not_found_on_error
def ping(model):
    model = ScoringService.get_model(model)
    status = 200
    res = {
        'model_status': {
            'state': 'AVAILABLE',
            'status': {'error_code': 'OK', 'error_message': ''},
        }
    }
    return res, status


@app.route(
    '/v1/models/<model>/metadata',
    methods=['GET']
)
@not_found_on_error
def metadata(model):
    metadata = ScoringService.metadata(model)
    assert metadata is not None, f'model {model} returned empty metadata'
    return metadata, 200


@app.route(
    '/v1/models/<model>:summarize',
    methods=['POST']
)
@not_found_on_error
def summarize(model):
    payload = flask.request.json

    assert ('text_data' in payload)
    data = payload['text_data']['data']

    # Do the prediction
    return ScoringService.summarize(model, data), 200


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return flask.Response(
        response='Model server is running!',
        status=200,
        mimetype='text/html'
    )
