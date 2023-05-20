from firebase_admin import storage
from datetime import datetime, timedelta


class StorageService:
    def __init__(self):
        self.bucket = storage.bucket()

    def upload_file(self, file_path, file_name):
        blob = self.bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        # Return the public url of the uploaded file
        return blob.public_url

    def download_file(self, file_name):
        blob = self.bucket.blob(file_name)
        if blob.exists():
            return blob.public_url
        else:
            return None

    def delete_file(self, file_name):
        blob = self.bucket.blob(file_name)
        blob.delete()
        # Return a success message
        return {'status': 'success'}

    def get_file_url(self, file_name):
        blob = self.bucket.blob(file_name)
        if blob.exists():
            # Set the expiration time for the URL
            expiration_time = datetime.now() + timedelta(hours=69)
            # Generate the signed URL
            url = blob.generate_signed_url(expiration=expiration_time)
            return url
        else:
            return None
