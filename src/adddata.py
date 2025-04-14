import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials
from firebase_admin import db
import os


load_dotenv()

cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
})

ref = db.reference('employees')

data = {
    "05032003":
    {
        "name": "Dang Anh Quan",
        "department": "Technical room",
        "starting_year": 2023,
        "sex": "male",
        "birthdate": "2003-03-05",
        "address": "Ha Noi",
        "last_login_time": "2025-01-01 12:00:00",
        "tatol_attendance": 7
    }
}

for key, value in data.items():
    ref.child(key).set(value)