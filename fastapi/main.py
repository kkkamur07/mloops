from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from http import HTTPStatus
from enum import Enum
import re
import cv2
from fastapi.responses import FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Welcome to the MNIST model inference API!")
    yield
    print("Goodbye")


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI(lifespan=lifespan)

@app.get("/") # Initial endpoint
def root():
    """health check"""
    response = {
        "message" : HTTPStatus.OK.phrase,
        "status-code" : HTTPStatus.OK
    }
    return response

# We can use this to restrict the items of choice
@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}

database = {'username': [ ], 'password': [ ]}

@app.post("/login/") #link would be http://localhost:8000/login/username=yourname&password=yourpassword
def login(username: str, password: str):
    """login endpoint"""

    if username in database['username']:

        return {"message": "User already exists"}

    else:
        database['username'].append(username)
        database['password'].append(password)

        with open('database.txt', 'a') as f:
            f.write(f"{database['username']}:{database['password']}\n")

        return {"message": "User created successfully", "username": username}

@app.get("/password/{username}")
def get_user(username: str):
    """get user endpoint"""

    if username in database['username']:
        index = database['username'].index(username)
        return {"username": username, "password": database['password'][index]}

    else:
        return {"message": "User not found", "status-code": HTTPStatus.NOT_FOUND}

@app.get("/email/{email}")
def get_email(email: str):
    """get email endpoint"""
    match = re.search(r"(?<=@)[^.]+(?=\.)", email)
    if match:
        domain = match.group(0)
        return {"email": email, "domain": domain}
    else:
        return {"message": "Invalid email format", "status-code": HTTPStatus.BAD_REQUEST}


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), w: int = 100, h: int = 100):
    # Save uploaded image
    contents = await data.read()
    with open("image.jpg", "wb") as f:
        f.write(contents)

    # Read and process image
    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (w, h))
    cv2.imwrite("image_resize.jpg", res)

    # Return resized image file
    return FileResponse("image_resize.jpg")
