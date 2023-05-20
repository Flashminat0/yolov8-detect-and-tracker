import json
from firebase_admin import credentials, db, initialize_app

# Load appsettings JSON file
with open('appsettings.json', 'r') as json_file:
    appsettings = json.load(json_file)

# Firebase-APIKey File
API_KEY_PATH = "firebase-api-key.json"  # Add your API file path

# Initialize the default firebase app
certificate = credentials.Certificate(API_KEY_PATH)
firebaseApp = initialize_app(certificate, {'databaseURL': appsettings['DatabaseURL']})


class ToDoCollection:
    def __init__(self):
        self.collection = db.reference(appsettings['TodoCollection'])
        self.key = appsettings['TodoCollectionUniqueKey']

    def __getSnapshot(self):
        return self.collection.get()

    def __findItem(self, id):
        snapshot = self.__getSnapshot()
        if snapshot is None:
            return False
        for key, val in snapshot.items():
            if val[self.key] == id:
                return self.collection.child(key)
        return False

    def addTodoItem(self, content):
        if self.key in content:
            if not self.__findItem(content[self.key]):
                self.collection.push(content)
                return True
            else:
                raise Exception(f"Item with id {content[self.key]} already exists")
        else:
            raise Exception(f"Key {self.key} not found in the content")

    def getTodoItems(self):
        snapshot = self.__getSnapshot()
        return [val for val in snapshot.values()] if snapshot else []

    def getTodoItem(self, id):
        todoList = self.getTodoItems()
        return next((item for item in todoList if item[self.key] == id), None)

    def clearAllItems(self):
        self.collection.delete()
        return True

    def updateTodoItem(self, id, content):
        itemMatchedNode = self.__findItem(id)
        if itemMatchedNode is False:
            raise Exception(f"Item with id {id} doesn't exists")
        itemMatchedNode.set(content)
        return True

    def deleteTodoItem(self, id):
        itemMatchedNode = self.__findItem(id)
        if itemMatchedNode is False:
            raise Exception(f"Item with id {id} doesn't exists")
        itemMatchedNode.delete()
        return True
