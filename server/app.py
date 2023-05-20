from firebase_service import ToDoCollection
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS, cross_origin

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

if __name__ == "__main__":
    todo = ToDoCollection()
    app.run(debug=True)  # Make sure debug is false on production environment
