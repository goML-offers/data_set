from fastapi import APIRouter, UploadFile, Header, Form, File, UploadFile, Body, BackgroundTasks
from app.services.asset_creation import asset_creation
from app.schemas.schema import assetCreation
from typing import List
import os
import logging
import multiprocessing
from datetime import datetime
import os
import signal
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')




logger = logging.getLogger(__name__)
log_file_path = f'/app/app/logs/asset_{timestamp}.log'
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


app = APIRouter()
# segmentation_processes = {}


def success_response(data, message="Data retrieved successfully"):
    return {
        "status": "success",
        "message": message,
        "data": data
    }

def error_response(message, status="error"):
    return {
        "status": status,
        "message": message
    }




@app.post("/api/asset/asset_creation/")
async def asset_data(data: assetCreation, background_tasks: BackgroundTasks):
    try:
        project_id = data.project_id
        user_id = data.user_id
        # parameters = data.parameters
        table_name = data.table_name
        file_id_logo = data.file_id_logo
        file_id_picture_bank_image = data.file_id_picture_bank_image
        extraction_id_brand_image = data.extraction_id_brand_document
        extraction_id_tone_of_voice = data.extraction_id_tone_of_voice


        
        # api_response = {"project_id": data.project_id, "user_id": data.user_id, "segment_id":data.segment_id, "file_group":data.file_group}


        background_tasks.add_task(asset_creation, table_name, user_id, project_id, extraction_id_brand_image, extraction_id_tone_of_voice, file_id_logo, file_id_picture_bank_image)


        return "Marketing & Communication asset API creation started."

    except Exception as e:
        return error_response(str(e))

def check_segmentation_status(segment_id, user_id, project_id):
    # Implement your logic to check the status (replace this with your real logic)
    # This example returns a completed status with table_name and segment_id
    return {"status": "completed", "table_name": f"segment_{user_id}_{project_id}", "segment_id": segment_id}




# Function to send email with image attachment
@app.post("/api/asset/send-image")
async def send_email_with_image(to_email, subject, image_path,filename):

    response = send_email_with_image(to_email, subject, image_path,filename)

    return response


