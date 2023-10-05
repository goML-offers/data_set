from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from supabase import Client, create_client
import os
import json
from boto3.session import Session
import logging
from dotenv import load_dotenv, find_dotenv
import openai
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


load_dotenv()



# logging.basicConfig(
#     level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.FileHandler(f'/app/logs/data_extraction_{timestamp}.log'),  # Log to a file
#         logging.StreamHandler()  # Log to the console
#     ]
# )

# logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)
log_file_path = f'/app/logs/data_extraction_{timestamp}.log'
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_bucket = 'solarplexus'
jwt_secret = "mWoXpkmrf6ClIyZQt8qX1Z4kUrKGa1vToqv7+NxQjIe7AcdRTioXmXSuhvwkvbvbLn03R1D6CZ+/qTuVQpSPTw=="

session = Session(aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
s3_session = session.resource('s3')
s3_folder = s3_session.Bucket('gomloffers')

# --------> user will upload file with the file type, that will be stored in the s3 and table will be updated with file type, file path and user id
# --------> insert of predefined questions for a particular file type will be done once we fetch the file type from the user
# --------> for document extraction fetch the pre defined questions based on file type and user id
# --------> once data extracted will be stored in corresponding questions row

try:

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info('Supabase connection successfull')

except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    logger.error(e)


def extraction(user_id, file_id):
    try:
        process_data_insert = [
                    {
                        "user_id" :user_id,
                        "process_type": "data_extraction",
                        "process_status": "in_progress",
                        "start_at" : datetime.now().isoformat()
                    },
                ]
                

        process= supabase.from_("process").insert(process_data_insert).execute()
        process_data = [record for record in process.data]

        p_data = json.dumps(process_data)
        p = json.loads(p_data)
        process_id = p[0]["process_id"]

        user = supabase.from_("file_data").select("*").eq("id",file_id).execute()

        user_data = [record for record in user.data]
        print(user_data)
        data = json.dumps(user_data)
        d = json.loads(data)

        file_path = d[0]["path"]
        file_type = d[0]["type"]
        file_language = d[0]["language"]

        columns = 'question_id', 'pre_defined_questions'
        # file_type = "brand_guidelines"
        # print("file typeeeeeeeee",file_type)
        quest = supabase.from_("questions").select(*columns).eq("file_type", file_type).execute()
        print(quest)
        quest_data = [record for record in quest.data]
        ques_data = json.dumps(quest_data)
        questions = json.loads(ques_data)

        print(questions)
        # question = {}
        list_of_dictionary_ques_ans = []
        for ques_details in questions:
            print(ques_details)
            question_dict = json.loads(ques_details['pre_defined_questions'])
            # print("question_dict-----------",question_dict)
            predefined_question = question_dict["questions"][0]

            # print("predefined_question----------",predefined_question)

            question_id = ques_details['question_id']
            print(question_id)

            

            # data_to_insert = [
            #     {
            #         "user_id": user_id,
            #         "file_id": file_id,
            #         "question_id": questions[0]['question_id']
            #     },
            # ]
            
            # response_insert = supabase.from_("data_extraction").upsert(data_to_insert).execute()

            # extraction_d = [record for record in response_insert.data]
            # response_i = json.dumps(extraction_d)
            # extraction_response = json.loads(response_i)
            # extraction_id = extraction_response[0]['extraction_id']

            # extraction_data = supabase.from_("data_extraction").select("*").eq("extraction_id", extraction_id).execute()

            # e = [record for record in extraction_data.data]
            # response_e = json.dumps(e)
            # e_response = json.loads(response_e)


            try:


                local_file_path = f'/app/uploads/{file_path.split("/")[-1]}' 
                print(local_file_path)
                # s3_folder.download_file(Key=file_path, Filename=local_file_path)
                
                print(file_path)
                with open(local_file_path, 'wb+') as f:
                    data = supabase.storage.from_(supabase_bucket).download(file_path)
                    f.write(data)



            except Exception as e:

                logging.error('An error occurred:', exc_info=True)


            reader = PdfReader(local_file_path)
            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text

            print(raw_text[:100])

            # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

            text_splitter = CharacterTextSplitter(        
                separator = "\n",
                chunk_size = 1000,
                chunk_overlap  = 200,
                length_function = len,
            )
            texts = text_splitter.split_text(raw_text)

            # Download embeddings from OpenAI
            embeddings = OpenAIEmbeddings()

            docsearch = FAISS.from_texts(texts, embeddings)

            if file_language == 'swedish':

                chain = load_qa_chain(OpenAI(), chain_type="stuff")

                llm_answer_list = []

                ques = predefined_question + "convert the result to swedish language and strictly don't give me anything in english"
                docs = docsearch.similarity_search(ques)
                llm_answer = chain.run(input_documents=docs, question=ques)

                question = {"question" : predefined_question, "answer": llm_answer}
                list_of_dictionary_ques_ans.append(question)
 



                    # data_to_insert = [
                    #     {
                    #         "user_id": user_id,
                    #         "file_id": file_id,
                    #         "question_id": questions[0]['question_id'],
                    #         "llm_answer": llm_answer
                    #     },
                    # ]
                    

                    # response_insert = supabase.from_("data_extraction").upsert(data_to_insert).execute()

                    # extraction_d = [record for record in response_insert.data]
                    # response_i = json.dumps(extraction_d)
                    # extraction_response = json.loads(response_i)
                    # extraction_id = extraction_response[0]['extraction_id']

            else:

                chain = load_qa_chain(OpenAI(), chain_type="stuff")

                llm_answer_list = []
                extracted_id_list = []

                docs = docsearch.similarity_search(predefined_question)
                llm_answer = chain.run(input_documents=docs, question=predefined_question)


                question = {"question" : predefined_question, "answer": llm_answer}
                list_of_dictionary_ques_ans.append(question)
                


        print(list_of_dictionary_ques_ans)

        data_to_insert = [
            {
                "user_id": user_id,
                "file_id": file_id,
                "llm_answer": question
            },
        ]
        

        response_insert = supabase.from_("data_extraction").upsert(data_to_insert).execute()

        extraction_d = [record for record in response_insert.data]
        response_i = json.dumps(extraction_d)
        extraction_response = json.loads(response_i)
        extraction_id = extraction_response[0]['extraction_id']
        # extracted_id_list.append(extraction_id)

        # llm_answer_dict= {"llm_generated_answers": llm_answer_list}
        # llm_answer_json = json.dumps(llm_answer_dict)


        # table = "data_extraction"
        # updates = {
        #     "llm_answer": llm_answer_json
        # }
        # response_update = supabase.from_(table).update(updates).eq("extraction_id", extraction_id).execute()

        # update_d = [record for record in response_update.data]
        # response_u = json.dumps(update_d)
        # update_response = json.loads(response_u)
        
        process_data_update = {
                        "process_status": "stopped",
                        "end_at" : datetime.now().isoformat()
                    }
        supabase.from_("process").update(process_data_update).eq("process_id", process_id).execute()
        
        os.remove(local_file_path)
        logger.info("extraction done") 

        # return {"extraction_id" :extracted_id_list, "llm_answer": llm_answer_json} 
        return {"extraction_id" :extraction_id, "question_answer": list_of_dictionary_ques_ans} 
    
    except Exception as e:
        logger.error(e)
        return {"error": e}

# user_id = 15
# file_id = 24
# extraction(user_id, file_id)



def update_answer(extraction_id, question_id, updated_answer):
    try: 
        table = "data_extraction"
        answer = {}
        answer[question_id] = updated_answer
        updates = {
            # "user_answer": updated_answer
            "user_answer": answer
        }
        response_update = supabase.from_(table).update(updates).eq("extraction_id", extraction_id).execute()

        update_d = [record for record in response_update.data]
        response_u = json.dumps(update_d)
        update_response = json.loads(response_u)
        extraction_id_update = update_response[0]["extraction_id"]
        
        logger.info("answer update done")

        return {"extraction_id" :extraction_id_update}
    
    except Exception as e:
        logger.error(e)
        return {"error": e}

# updated_answer = "the colour is blue"
# extraction_id = 9
# update_answer(extraction_id, updated_answer)


def insert_file_data(file_path, file_type, file_language, user_id):
    try: 
        data_to_insert = [
            {
                "file_path": file_path,
                "file_type": file_type,
                "file_language": file_language,
                "user_id": user_id
            },
        ]
        response_insert = supabase.from_("file_data").upsert(data_to_insert).execute()

        extraction_d = [record for record in response_insert.data]
        response_i = json.dumps(extraction_d)
        file_response = json.loads(response_i)
        file_id = file_response[0]['sid']

        logger.info("inserted file data done")

        return {"file_id", file_id}
    except Exception as e:
        logger.error(e)
        return {"error": e}

# user_id = 3
# file_path = "uploads/Solarplexus.ai Brand guideline_final 230731 (1).pdf"
# file_type = "Brand Guidelines"
# file_language = "English"
# insert_file_data(file_path, file_type, file_language, user_id)


def create_supabase_bucket(file_name,bucket_path):
    # print("Bucket Pathhhhhhhhhhhhhhh", bucket_path)
    with open(file_name, 'rb') as f:
        supabase.storage.from_(supabase_bucket).upload(file=f,path=bucket_path)

    return f'Successfully uploaded {file_name} to {bucket_path}'
