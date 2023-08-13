import os
import json
from firebase_admin import credentials, initialize_app

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

# functions
from capture_to_find import capture_to_find

# controllers
from controllers.notifications import NotificationCollection
from controllers.storage_service import StorageService
from controllers.todos import ToDoCollection

app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)

# Load appsettings JSON file
appsettings = None
with open('controllers/appsettings.json', 'r') as json_file:
    appsettings = json.load(json_file)

# Firebase-APIKey File
API_KEY_PATH = "controllers/firebase-api-key.json"

# Initialize the default firebase app
certificate = credentials.Certificate(API_KEY_PATH)
firebaseApp = initialize_app(certificate, {
    'databaseURL': appsettings['DatabaseURL'],
    'storageBucket': appsettings['StorageURL'],
})


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
            url = storage.get_file_url(image_name)
            return jsonify({'url': url})
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


class CompareImages(Resource):
    def post(self):
        try:
            if 'image_from_phone' not in request.files:
                return 'No image_from_phone in request', 400

            image_from_app_name = request.files['image_from_phone']

            user = request.form.get('user')
            image_2_frame = request.form.get('frame')
            image_2_class_idx = request.form.get('class_idx')

            if not all([user, image_2_frame, image_2_class_idx]):
                return 'Missing data in request', 400

            file_extension = image_from_app_name.filename.split('.')[-1]

            image_name = secure_filename(
                user + '_' + image_2_class_idx + '.' + file_extension)
            image_path = os.path.join('tmp', image_name)

            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            image_from_app_name.save(image_path)

            # save image from app to storage
            storage = StorageService()
            response = storage.upload_file(image_path, image_name)  # Corrected this line

            # After upload, delete the temporary local file
            os.remove(image_path)

            return jsonify(response)

        except Exception as ex:
            return str(ex), 400


api.add_resource(CompareImages, '/compareImages')


class FindTheLaptop(Resource):
    def post(self):
        try:
            json_data = request.get_json(force=True)
            image_url = json_data['image_url']
            user = json_data['user']
            id_job = json_data['id_job']

            data = capture_to_find(image_url, user, id_job)

            return jsonify(data)

        except Exception as ex:
            return str(ex), 400


api.add_resource(FindTheLaptop, '/findTheLaptop')

if __name__ == "__main__":
    todo = ToDoCollection()
    notification = NotificationCollection()
    app.run(debug=True)  # Make sure debug is false on production environment
