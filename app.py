import os
import json
import shutil
from datetime import datetime

from firebase_admin import credentials, initialize_app

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

from capture_thieves import capture_to_find_thieves
# functions
from capture_to_find import capture_to_find
from capture_to_find_v2 import capture_to_find_v2
from check_similarity import image_similarity

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
            image_path = os.path.join('task', image_name)

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
            time_stamp = request.form.get('frame')
            image_2_class_idx = request.form.get('class_idx')

            if not all([user, time_stamp, image_2_class_idx]):
                missing = []
                if not user:
                    missing.append('user')
                if not time_stamp:
                    missing.append('frame')
                if not image_2_class_idx:
                    missing.append('class_idx')
                return f'Missing fields: {", ".join(missing)}', 400

            file_extension = image_from_app_name.filename.split('.')[-1]

            image_name = secure_filename(
                user + '.' + file_extension)
            image_path = os.path.join('task', image_name)

            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            image_from_app_name.save(image_path)

            data = capture_to_find_v2(user)

            mobile_image_path = f'task/{user}.jpg'
            laptop_images = data['laptops']

            frame_image_path = data['frame']

            laptops_with_similarities = []
            for laptop_image in laptop_images:
                laptop_image_to_check = f'task/{user}_laptop_{laptop_image["laptopID"]}.jpg'

                similarity_score = image_similarity(mobile_image_path, laptop_image_to_check)

                checked_laptop = {
                    'laptop_image_path': laptop_image_to_check,
                    'similarity_score': float(similarity_score),
                    'coordinates': laptop_image['coordinates']
                }

                laptops_with_similarities.append(checked_laptop)

            laptops_with_similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

            # laptop with highest similarity score
            laptop_with_highest_similarity = laptops_with_similarities[0]

            storage = StorageService()
            response = storage.upload_file(laptop_with_highest_similarity['laptop_image_path'],
                                           f'jobs/{user}/{time_stamp}/laptop_image.jpg')
            response2 = storage.upload_file(frame_image_path, f'jobs/{user}/{time_stamp}/frame_image.jpg')

            return json.dumps(laptop_with_highest_similarity, indent=4)

        except Exception as ex:
            return str(ex), 400


api.add_resource(CompareImages, '/compareImages')


class Restart(Resource):
    def post(self):
        # Remove the last line from the file
        with open(__file__, 'r') as f:
            lines = f.readlines()
        with open(__file__, 'w') as f:
            f.writelines(lines[:-1])

        # Touch the restart file to trigger the reloader
        with open(__file__, 'a') as f:
            f.write(f"\n# Restarted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "Server is restarting..."


api.add_resource(Restart, '/restart')


class FindThieves(Resource):
    def post(self):
        try:
            json_data = request.get_json(force=True)

            user = json_data['user']
            x_1 = json_data['x1']
            x_2 = json_data['x2']
            y_1 = json_data['y1']
            y_2 = json_data['y2']
            time_stamp = json_data['timestamp']

            data = capture_to_find_thieves()

            storage = StorageService()

            image_dict = {}

            print('uploading images ...\n\n')
            for image_url in data:
                # extract image name from path
                img_name = os.path.basename(image_url)

                # extract ID from image name
                _, img_id, _ = img_name.split('_')

                # upload to Firebase with the new path structure
                firebase_path = f'jobs/{user}/{time_stamp}/thieves/{img_id}/{img_name}'
                upload_response = storage.upload_file(image_url, firebase_path)

                # get URL of uploaded image
                img_url = storage.get_file_url(firebase_path)

                if img_id not in image_dict:
                    image_dict[img_id] = []
                image_dict[img_id].append(img_url)

                print(f'\ruploadeding images: {len(sum(image_dict.values(), [])) / len(data) * 100:.2f}%', end='')

            print('\nuploading done ...\n')

            response_data = [{'id': int(key), 'images': value} for key, value in image_dict.items()]

            return jsonify(response_data)

        except Exception as ex:
            return str(ex), 400


api.add_resource(FindThieves, '/findThieves')


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


class SendNotification(Resource):
    def post(self):
        try:
            json_data = request.get_json(force=True)
            expo_token = json_data['to']
            title = json_data['title']
            body = json_data['body']

            url = "https://exp.host/--/api/v2/push/send"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "to": expo_token,
                "title": title,
                "body": body
            }

            response = requests.post(url, json=payload, headers=headers)
            return response.json(), response.status_code

        except Exception as ex:
            return str(ex), 400


api.add_resource(SendNotification, '/notification/')

if __name__ == "__main__":
    todo = ToDoCollection()
    notification = NotificationCollection()
    app.run(debug=True, use_reloader=True)  # Make sure debug is false on production environment



# Restarted at 2023-10-30 12:49:14