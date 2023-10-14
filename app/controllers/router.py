from fastapi import APIRouter, UploadFile, Header, Form, File, UploadFile, Body
from app.services.segmentation import segmentation
from app.schemas.schema import segementcreation
from typing import List
import os
import logging
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')




logger = logging.getLogger(__name__)
log_file_path = f'/app/app/logs/segmentation_{timestamp}.log'
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


app = APIRouter()

# @app.post("/upload/")
# async def upload_file(
#     user_id: int = Header(..., convert_underscores=False),  # Get user_id from header
#     file_type: str = Form(...),
#     file: UploadFile = File(...),):

#     #upload file in s3 bucket
#     #table insert the s3 path, file_name, user_name
#     "push to s3"
#     "insert into table"
#     filepath = "s3 path"
#     return {"user_id": user_id, "file_path": filepath , "file_type" :file_type}





# @app.post("/api/brand_guidelines_asset/upload_file/")
# async def upload_files(
#     files: List[UploadFile] = File(...),
#     user_id: str = Body(...),
#     file_type: str = Body(...),
#     file_language: str = Body(...)
#     # file_path: str = Body(...)
# ):

#     try:

#         uploaded_file_paths = []
#         # folder_name = 'solarplexus/brand-guidelines-asset'
#         folder_name = '/brand-guidelines-asset'
#         print(folder_name)
#         subfolder_name = f'{user_id}/{file_type}'
#         print(subfolder_name)
#         # file_name = file_path

#         # files_name = file_path.split("/")[-1]
#         # file_name = file.filename
#         # bucket_path = f"{folder_name}/{subfolder_name}/{files_name}"

#         file_id = None
#         UPLOAD_DIR = "uploads"

#         if not os.path.exists(UPLOAD_DIR):
#             os.makedirs(UPLOAD_DIR)
#         file_path_list = []
#         file_id_list = []
#         for file in files:
#             bucket_path = f"{folder_name}/{subfolder_name}/{file.filename}"
#             print(bucket_path)
#             file_p = os.path.join(UPLOAD_DIR, file.filename)
#             with open(file_p, "wb") as f:
#                 f.write(file.file.read())
#             uploaded_file_paths.append(file_p)

#             # print(UPLOAD_DIR)

#             # res = supabase.storage.create_bucket(name)

#             # s3.upload_file(
#             #     file_name,
#             #     BUCKET_NAME,
#             #     f'{folder_name}/{subfolder_name}/{file_name}'
#             # )

#             uploaded_file = create_supabase_bucket(file_p,bucket_path)
#             print(uploaded_file)


#             # print(f'Successfully uploaded {file_name} to {BUCKET_NAME}/{folder_name}/{subfolder_name}/')

#             s3_file_path = f"{folder_name}/{subfolder_name}/{file.filename}"
#             file_path_list.append(s3_file_path)
#             file_id = insert_file_data(s3_file_path, file_type, file_language, user_id)
#             file_id_list.append(file_id)
#             # doc = create_supabase_bucket(file_path, user_id, file_type)

#         os.remove(file_p)


#         logger.info("inserted file data done")

#         return {"file_paths": uploaded_file_paths, "file_path":f"{folder_name}/{subfolder_name}/{file.filename}", "user_id" : user_id , "file_type": file_type , "file_language": file_language , "file_id" : file_id_list, "files": file_path_list}

#     except Exception as e:
#         print(e)

@app.post("/api/segmentation/segment_creation/")
async def segment_data(data: segementcreation):
    try:
        project_id = data.project_id
        user_id = data.user_id
        parameters = data.parameters
        feature_selection = data.feature_selection

        segment = segmentation(project_id, user_id, parameters, feature_selection)

        logger.info("segmentation done")

        return segment

    except Exception as e:
        logger.error(e)
        return {"error": e}


# @app.put("/api/brand_guidelines_asset/update_answer/")
# async def update_extraction(data: UpdateExtractionSchema):
#     try:
#         extract_id = data.extraction_id
#         user_answer = data.user_answer
#         extraction_id = update_answer(extract_id, user_answer)

#         logger.info("update extracted data done")

#         return {"message": f"answer updated with ID {extraction_id} updated successfully."}

#     except Exception as e:
#         logger.error(e)
#         return {"error": e}
