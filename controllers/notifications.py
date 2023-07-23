import json
from firebase_admin import credentials, db, initialize_app

with open('controllers/appsettings.json', 'r') as json_file:
    appsettings = json.load(json_file)


class NotificationCollection:
    def __init__(self):
        self.collection = db.reference(appsettings['NotificationCollection'])
        self.key = appsettings['NotificationCollectionUniqueKey']

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

    def addNotification(self, content):
        if self.key in content:
            if not self.__findItem(content[self.key]):
                self.collection.push(content)
                return True
            else:
                raise Exception(f"Item with id {content[self.key]} already exists")
        else:
            raise Exception(f"Key {self.key} not found in the content")

    def getNotifications(self):
        snapshot = self.__getSnapshot()
        return [val for val in snapshot.values()] if snapshot else []

    def getNotification(self, id):
        todoList = self.getNotifications()
        return next((item for item in todoList if item[self.key] == id), None)

    def clearAllNotifications(self):
        self.collection.delete()
        return True

    def updateNotificationItem(self, id, content):
        itemMatchedNode = self.__findItem(id)
        if itemMatchedNode is False:
            raise Exception(f"Item with id {id} doesn't exists")
        itemMatchedNode.set(content)
        return True

    def deleteNotificationItem(self, id):
        itemMatchedNode = self.__findItem(id)
        if itemMatchedNode is False:
            raise Exception(f"Item with id {id} doesn't exists")
        itemMatchedNode.delete()
        return True
