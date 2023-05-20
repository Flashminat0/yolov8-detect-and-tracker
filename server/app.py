import os

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

from firebase_service import ToDoCollection
from storage_service import StorageService

app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)


class HealthCheck(Resource):
    def get(self):
        return "Healthy"


api.add_resource(HealthCheck, '/health')


class TodoItemsList(Resource):
    def get(self):
        try:
            todoItems = todo.getTodoItems()
            return jsonify(todoItems)
        except Exception as ex:
            return str(ex), 400


api.add_resource(TodoItemsList, '/getAllItems')


class GetTodoItem(Resource):
    def post(self):
        try:
            json_data = request.get_json(force=True)
            itemId = json_data['id']
            itemValue = todo.getTodoItem(itemId)
            return jsonify(itemValue)
        except Exception as ex:
            return str(ex), 400


api.add_resource(GetTodoItem, '/getItem')


class AddToDoItem(Resource):
    def post(self):
        try:
            json_data = request.get_json(force=True)
            itemValue = todo.addTodoItem(json_data)
            return jsonify(itemValue)
        except Exception as ex:
            return str(ex), 400


api.add_resource(AddToDoItem, '/addItem')


class DeleteToDoItem(Resource):
    def delete(self):
        try:
            json_data = request.get_json(force=True)
            itemId = json_data['id']
            status = todo.deleteTodoItem(itemId)
            return jsonify(status)
        except Exception as ex:
            return str(ex), 400


api.add_resource(DeleteToDoItem, '/deleteItem')


class DeleteAllToDoItems(Resource):
    def delete(self):
        try:
            status = todo.clearAllItems()
            return jsonify(status)
        except Exception as ex:
            return str(ex), 400


api.add_resource(DeleteAllToDoItems, '/deleteAllItems')


class UpdateToDoItem(Resource):
    def post(self):
        try:
            json_data = request.get_json(force=True)
            itemId = json_data['id']
            status = todo.updateTodoItem(itemId, json_data)
            return jsonify(status)
        except Exception as ex:
            return str(ex), 400


api.add_resource(UpdateToDoItem, '/updateItem')


class UploadImage(Resource):
    def post(self):
        try:
            if 'image' not in request.files:
                return 'No image file in request', 400
            image = request.files['image']

            image_name = secure_filename(image.filename)
            image_path = os.path.join('tmp', image_name)

            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            image.save(image_path)

            storage = StorageService()
            response = storage.upload_file(image_path, image_name)

            # After upload, delete the temporary local file
            os.remove(image_path)

            return jsonify(response)
        except Exception as ex:
            return str(ex), 400


api.add_resource(UploadImage, '/uploadImage')


class DownloadImage(Resource):
    def get(self, image_name):
        try:
            storage = StorageService()
            response = storage.download_file(image_name)
            return send_file(response, as_attachment=True)
        except Exception as ex:
            return str(ex), 400


api.add_resource(DownloadImage, '/downloadImage/<string:image_name>')


class DeleteImage(Resource):
    def delete(self, image_name):
        try:
            storage = StorageService()
            response = storage.delete_file(image_name)
            return jsonify(response)
        except Exception as ex:
            return str(ex), 400


api.add_resource(DeleteImage, '/deleteImage/<string:image_name>')

if __name__ == "__main__":
    todo = ToDoCollection()
    app.run(debug=True)  # Make sure debug is false on production environment
