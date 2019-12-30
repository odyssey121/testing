import requests
import json
from os import getcwd, mkdir,getenv,pardir
from os.path import join,isdir,dirname,abspath
from dotenv import load_dotenv

rootDir = dirname(__file__)
dotEnvPath = abspath(join(rootDir, '.env'))
load_dotenv(dotEnvPath)



class ApiInteraction:

    HEADERS = {
        'Content-Type': 'application/json',
    }


    def __init__(self):
        self.get_token()

    def get_token(self):
        token = None
        url = 'https://cropio.com/api/v3/sign_in'
        json_data = {
            "user_login": {
                "email":getenv("EMAIL"),
                "password":getenv("PASSWORD")

            }
        }
        response = requests.post(url, json=json_data, headers=self.HEADERS)
        if response.status_code == 200 and response.json()['success']:
            token = response.json().get("user_api_token", None)
            self.HEADERS['X-User-Api-Token'] = token

    def get_photo(self):
        url = 'https://cropio.com/api/v3/photos'
        response = requests.get(url, headers=self.HEADERS)
        if response.status_code == 200 and response.json()["data"]:
            photo_list = list()
            for item in response.json()["data"]:
                photo = item.get('photo',None)
                if photo:
                    photo_list.append(photo['photo']['url'])
            return photo_list

    def save_photos(self, pathToSave=join(getcwd(),'all_photo')):
        if not isdir(pathToSave): mkdir(pathToSave)
        for photo_url in self.get_photo():
            response = requests.get(photo_url)
            filename = response.url.split('/')[-2]
            with open(f"{join(pathToSave,filename)}.jpg", 'wb') as file:
                file.write(response.content)



if __name__ == '__main__':
    api = ApiInteraction()
    print(api.get_photo())
