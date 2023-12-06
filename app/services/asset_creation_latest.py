from supabase import Client, create_client
import pandas as pd
import json
import numpy as np
import openai 
# from openai import OpenAI
#from diffusers import DiffusionPipeline
#import torch
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import OpenAIModerationChain, SequentialChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from datetime import datetime
import logging
from dotenv import load_dotenv, find_dotenv
import together
import base64
import requests
import ast

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

load_dotenv()


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



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_bucket = 'solarplexus'
SUPABASE_HOST= os.environ.get("SUPABASE_HOST")
SUPABASE_PASSWORD= os.environ.get("SUPABASE_PASSWORD")
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
together.api_key = TOGETHER_API_KEY


try:

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info('Supabase connection successfull')

    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16")
    # commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")


except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    logger.error(e)


def fetch_table_data(table_name):
    table_response = supabase.from_(table_name).select("*").execute()
    table_data_raw = [record for record in table_response.data]

    # print(table_data_raw)

    table_data_json = json.dumps(table_data_raw)
    table= json.loads(table_data_json)
    # update_segment_id = update[0]['id']
    # print(update_segment_id)

    return table


def generate_text(cluster_description, categorical_description, asset_to_individulize_file, tone_of_voice):

    print("generate text")
    if asset_to_individulize_file is not None:
        print("asset to individualize")
        reader = PdfReader(asset_to_individulize_file)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # print(raw_text[:100])

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
        print("faiss")

        docsearch = FAISS.from_texts(texts, embeddings)
        print("-----faiss")


        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        llm_answer_list = []



        customer_profile = f"""I have a dataset with categorical and cluster descriptions and I need your help to understand
        and summarize the key characteristics of my audience for marketing purposes. Please analyze the following data and 
        provide a summary in 1 paragragph:
        {categorical_description} {cluster_description} Based on this information, please provide a comprehensive summary of the audience profile, 
        highlighting key aspects such as demographics, company characteristics, industry focus, technology usage, and any other notable trends. 
        This summary will assist in tailoring our marketing strategies effectively.
        (Just give summary no other information)"""


        profiling = openai.chat.completions.create(
            # model="gpt-3.5-turbo-0613",
            model= "gpt-4",
            # model = "gpt-3.5-turbo-16k",
            # model="gpt-4-0613",
            messages=[

                    {"role": "user",
                        "content": customer_profile},

                ]
            )
        customer_profile_summary = profiling.choices[0].message.content
        # customer_profile_summary = ''
        # for choice in profiling.choices:
        #     customer_profile_summary += choice.message.content

        print("customer_profile_summary--------",customer_profile_summary)
        print(type(customer_profile_summary))



    
        ques = f"""Create a short, engaging marketing text based on the provided customer profile summary: {customer_profile_summary}.
        Ensure the content resonates with the specific demographics, industry focus, and company characteristics outlined in the profile. 
        The tone of voice should be {tone_of_voice}, aligning with the preferences and expectations of the target audience.
        Highlight key benefits and solutions relevant to this unique customer base, emphasizing how our services meet their specific needs and goals.
        Also provide a two to three words headline as well based on the customer profile and the result you generated should only be in dictionary format strictly having two keys 
        summary and headline only having key and value enclosed in double quotes.
        follow the example for reference.
        for example:""" + """{"summary": "summary of the customer profile", "headline": "headline of the summary"}"""

        # ques = f"""convert the result to very short and attractive summary strictly based on the cluster description {cluster_description} and categorical data {categorical_description}, 
        #         make the results proper and different based only on the industry the cluster is focused and the answer should be very clear. The tone of the text
        #         should be in strictly {tone_of_voice} only. Don't give me anything else. The result should be attractive that can be used for marketing campaigns."""
        


        docs = docsearch.similarity_search(ques)
        llm_answer = chain.run(input_documents=docs, question=ques)

        print("llm_answer----------->", llm_answer)
        llm_json = json.loads(llm_answer)

        return llm_json
    
    else:
        
        customer_profile = f"""I have a dataset with categorical and cluster descriptions and I need your help to understand
        and summarize the key characteristics of my audience for marketing purposes. Please analyze the following data and 
        provide a summary in 1 paragragph:
        {categorical_description} {cluster_description} Based on this information, please provide a comprehensive summary of the audience profile, 
        highlighting key aspects such as demographics, company characteristics, industry focus, technology usage, and any other notable trends. 
        This summary will assist in tailoring our marketing strategies effectively.
        (Just give summary no other information)
        Also provide a two to three words headline as well based on the customer profile and the result you generated should only be in dictionary format strictly having two keys 
        summary and headline only having key and value enclosed in double quotes.
        follow the example for reference.
        for example:""" + """{"summary": "summary of the customer profile", "headline": "headline of the summary"}"""


        profiling = openai.chat.completions.create(
            # model="gpt-3.5-turbo-0613",
            model= "gpt-4",
            # model = "gpt-3.5-turbo-16k",
            # model="gpt-4-0613",
            messages=[

                    {"role": "user",
                        "content": customer_profile},

                ]
            )
        customer_profile_summary = profiling.choices[0].message.content
        # customer_profile_summary = ''
        # for choice in profiling.choices:
        #     customer_profile_summary += choice.message.content

        print("customer_profile_summary--------",customer_profile_summary)
        print(type(customer_profile_summary))

    return customer_profile_summary


def get_extracted_data(extraction_id_brand, extraction_id_tone):

    query = supabase.from_("data_extraction").select("*").eq("extraction_id", extraction_id_brand).execute()
    update_d = [record for record in query.data]

    print(update_d)
    color_data = json.dumps(update_d)
    color = json.loads(color_data)
    llm_answer = color[0]["llm_answer"]

    print(type(llm_answer))
    print(llm_answer)
    answer =  json.loads(llm_answer)

    # Create variables to store answers
    primary_color = None
    brand_name = None
    primary_font = None
    secondary_color = None
    secondary_font = None

    # Process the list of dictionaries
    for item in answer:
        question = item['question']
        answer = item['answer']

        if "primary colour code" in question:
            primary_color = answer
        elif "brand name" in question:
            brand_name = answer
        elif "primary font" in question:
            primary_font = answer
        elif "secondary colour code" in question:
            secondary_color = answer
        elif "secondary font" in question:
            secondary_font = answer

    # Print the stored answers
    print("Primary Color:", primary_color)
    print("Brand Name:", brand_name)
    print("Primary Font:", primary_font)
    print("Secondary Color:", secondary_color)
    print("Secondary Font:", secondary_font)


    response = supabase.from_("data_extraction").select("*").eq("extraction_id", extraction_id_tone).execute()
    response_d = [record for record in response.data]
    if response_d:
        print(response_d)
        tone_data = json.dumps(response_d)
        tone = json.loads(tone_data)
        tone_llm_answer = tone[0]["llm_answer"]

        print(type(tone_llm_answer))
        print(tone_llm_answer)
        tone_answer =  json.loads(tone_llm_answer)

        # Create variables to store answers
        tone_of_voice = None

        # Process the list of dictionaries
        for item in tone_answer:
            question = item['question']
            answer = item['answer']

            if "tone of voice" in question:
                tone_of_voice = answer

    else:
        tone_of_voice = "simple"

    # Print the stored answers
    print("tone of voice:", tone_of_voice)





    return {"primary_color": primary_color, "secondary_color": secondary_color, "primary_font": primary_font, "secondary_font":secondary_font, "brand_name": brand_name, "tone_of_voice": tone_of_voice}
    

def get_rgb_colors(primary_color, secondary_color):
    rgb_color = openai.chat.completions.create(
            # model="gpt-3.5-turbo-0613",
            model= "gpt-4",
            # model = "gpt-3.5-turbo-16k",
            # model="gpt-4-0613",
            messages=[

                    {"role": "user",
                        "content": f"""Generate RGB of color {primary_color} and color {secondary_color} and give me a json format strictly only in Red Green Blue nested dictionary and nothing else.
                                    You can consider this as an example to generate you result: 
                                    EXAMPLE: """ + """{"EB691B": { "Red": 235,"Green": 105"Blue": 27},"4B4B4B": { "Red": 75,"Green": 75,"Blue": 75},"95CDED": {"Red": 149,"Green": 205, "Blue": 237}}"""},

                ]
            )
    rgb_result = rgb_color.choices[0].message.content
    # rgb_result = ''
    # for choice in rgb_color.choices:
    #     rgb_result += choice.message.content

    print(rgb_result)
    print(type(rgb_result))
    
    
    
    "------------------------covert to json------------------------------"
    
    colors = json.loads(rgb_result)
    print(colors)
    print(type(colors))
    
    "------------------------reading rgb from json------------------------"
    

    # Initialize variables for primary and secondary colors
    primary_color_rgb = ()
    secondary_color_rgb = ()

    # Iterate through the dictionary and store RGB values for the first two keys
    for idx, (key, rgb_values) in enumerate(colors.items()):
        if idx == 0:
            primary_color_rgb = (rgb_values['Red'], rgb_values['Green'], rgb_values['Blue'])
        elif idx == 1:
            secondary_color_rgb = (rgb_values['Red'], rgb_values['Green'], rgb_values['Blue'])
        else:
            break  # Only store values for the first two keys

    # Print the stored RGB values
    print(f"Primary Color: {primary_color_rgb}")
    print(f"Secondary Color: {secondary_color_rgb}")  

    return {"primary_color_rgb": primary_color_rgb, "secondary_color_rgb": secondary_color_rgb}  


def fetch_background_image(file_id_background_image):
    type = "picture_bank"
    user = supabase.from_("file_data").select("*").eq("id",file_id_background_image).eq("type", type).execute()

    user_data = [record for record in user.data]
    print("user_data",user_data)
    data = json.dumps(user_data)
    d = json.loads(data)

    file_path = d[0]["path"]
    file_type = d[0]["type"]



    try:


        local_file_path = f'/app/app/services/{file_path.split("/")[-1]}'
        print(local_file_path)

        print(file_path)
        with open(local_file_path, 'wb+') as f:
            data = supabase.storage.from_(supabase_bucket).download(file_path)
            f.write(data)

    except Exception as e:

        logging.error('An error occurred:', exc_info=True)

    return local_file_path

# fetch_background_image(803)


    

def fetch_logo(file_id_log):
    type = "logo"
    user = supabase.from_("file_data").select("*").eq("id",file_id_log).eq("type", type).execute()

    user_data = [record for record in user.data]
    print("user_data",user_data)
    data = json.dumps(user_data)
    d = json.loads(data)

    file_path = d[0]["path"]
    file_type = d[0]["type"]


    try:


        local_file_path = f'/app/app/services/{file_path.split("/")[-1]}'
        print(local_file_path)

        print(file_path)
        with open(local_file_path, 'wb+') as f:
            data = supabase.storage.from_(supabase_bucket).download(file_path)
            f.write(data)

    except Exception as e:

        logging.error('An error occurred:', exc_info=True)

    return local_file_path


def fetch_asset_individualize(project_id):
    group = "asset"
    user = supabase.from_("project_files").select("*").eq("project_id",project_id).eq("group", group).execute()

    user_data = [record for record in user.data]

    if user_data:
        print("user_data",user_data)
        data = json.dumps(user_data)
        d = json.loads(data)

        file_path = d[0]["path"]
        file_group = d[0]["group"]


        try:


            local_file_path = f'/app/app/services/{file_path.split("/")[-1]}'
            print(local_file_path)

            print(file_path)
            with open(local_file_path, 'wb+') as f:
                data = supabase.storage.from_(supabase_bucket).download(file_path)
                f.write(data)

        except Exception as e:

            logging.error('An error occurred:', exc_info=True)

        return local_file_path
    else:
        return None    




def format_text(text, max_words_per_line):
    words = text.split()
    lines = [words[i:i + max_words_per_line] for i in range(0, len(words), max_words_per_line)]
    formatted_text = "\n".join([" ".join(line) for line in lines])
    return formatted_text

"""def combine_text_image(cluster_id, background_image_path, logo_path, asset_to_individualize, primary_color_rgb, secondary_color_rgb):
    print("combine text image function")
    font_size =30
    width, height = 500, 400
    base_image = Image.open(background_image_path)
    base_image = base_image.resize((width, height))

    # Initialize the drawing context
    draw = ImageDraw.Draw(base_image)

    # Set primary and secondary colors
    primary_color_rgb = primary_color_rgb  # (R, G, B) for #EB691B
    secondary_color_rgb = secondary_color_rgb  # (R, G, B) for #4B4B4B

    # Use the default font
    font = ImageFont.load_default()
    font = font.font_variant(size=font_size)

    # Set the text to be displayed
    formatted_text = asset_to_individualize['summary']
    headline = asset_to_individualize['headline']
    formatted_text = format_text(formatted_text, 10)  # Adjust the max_words_per_line as needed
    text = formatted_text

    # Set the text position for the primary color
    text_position_primary = (30, 100)
    # text_position_primary = (width // 2, height // 2)  # Center of the image


    # Draw text in primary color with default font and specified size
    draw.text(text_position_primary, text, fill=secondary_color_rgb, font=ImageFont.load_default())
    # Draw text with center alignment
    # draw.text(text_position_primary, text, fill=primary_color_rgb, font=font, anchor="mm")  # "mm" stands for middle-middle

    # Set the text position for the headline
    headline_position = (30, 50)

    # Draw headline in primary color
    draw.text(headline_position, headline, fill=primary_color_rgb, font=font)


    # Load the overlay image
    logo = Image.open(logo_path)

    # You may need to resize the overlay image to fit
    logo = logo.resize((80, 50))  # Adjust the size as needed

    # Paste the overlay image on top of the base image
    base_image.paste(logo, (400, 20))

    # Convert the image to RGB mode before saving
    base_image = base_image.convert("RGB")

    # Save the modified image
    asset_path = f"asset_{cluster_id}.jpg"
    base_image.save(asset_path)

    # Display the modified image
    # base_image.show()

    return asset_path"""



def draw_bold_text(draw, position, text, font, fill, iterations=2, offset=1):
    for _ in range(iterations):
        draw.text((position[0] + offset, position[1]), text, font=font, fill=fill)
        offset += 1

def combine_text_image(cluster_id, background_image_path, logo_path, asset_to_individualize, primary_color_rgb, secondary_color_rgb):
    print("combine text image function")
    headline_font_size =40
    font_size = 25  # Increase the font size
    width, height = 1024, 1024
    base_image = Image.open(background_image_path)
    base_image = base_image.resize((width, height))

    # Initialize the drawing context
    draw = ImageDraw.Draw(base_image)

    # Set primary and secondary colors
    primary_color_rgb = primary_color_rgb  # (R, G, B) for #EB691B
    secondary_color_rgb = secondary_color_rgb  # (R, G, B) for #4B4B4B

    # Use the default font
    default_font = ImageFont.load_default()

    # Set the text to be displayed
    formatted_text = asset_to_individualize['summary']
    headline = asset_to_individualize['headline']
    formatted_text = format_text(formatted_text, 10)  # Adjust the max_words_per_line as needed
    text = formatted_text

    # Set the text position for the primary color
    text_position_primary = (100, 500)

    # Draw bold text in primary color with larger font size
    draw_bold_text(draw, text_position_primary, text, default_font.font_variant(size=font_size), secondary_color_rgb)

    # Set the text position for the headline
    headline_position = (200, 350)

    # Draw bold headline in primary color with larger font size
    draw_bold_text(draw, headline_position, headline, default_font.font_variant(size=headline_font_size), primary_color_rgb)

    # Load the overlay image
    logo = Image.open(logo_path)

    # You may need to resize the overlay image to fit
    logo = logo.resize((80, 50))  # Adjust the size as needed

    # Paste the overlay image on top of the base image
    base_image.paste(logo, (900, 20))

    # Convert the image to RGB mode before saving
    base_image = base_image.convert("RGB")

    # Save the modified image
    asset_path = f"asset_{cluster_id}.jpg"
    base_image.save(asset_path)

    # Display the modified image
    # base_image.show()

    return asset_path





def marketing_image_creation(cluster_id, cluster_description, categorical_description, primary_color_rgb, secondary_color_rgb):
    prompt = f"""generate a marketing asset image for a brand , to target big industries, 
    image should have primary colour code should be {primary_color_rgb} and secondary color code should be {secondary_color_rgb}, and generate images based on these cluster
    description {cluster_description} and {categorical_description}"""
    image = pipe(prompt).images[0]
    print(image)
    filename = f'result_{cluster_id}.jpg'
    image.save(filename)
    print(filename)
    # result_filenames.append(filename)

    return filename






def CreateCustomerProfile(categorical_description, cluster_description):
    
    prompt = f"""I have a dataset with categorical and cluster descriptions and I need your help to understand and summarize the key characteristics of my audience for marketing purposes. Please analyze the following data and provide a summary in 1 paragragph:
                {categorical_description}
                {cluster_description} 
                Based on this information, please provide a comprehensive summary of the audience profile, highlighting key aspects such as demographics, company characteristics, industry focus, technology usage, and any other notable trends. This summary will assist in tailoring our marketing strategies effectively.
                (Just give summary no other information)
                Summary: """


    # Get the completion from the OpenAI API
    completion = openai.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "user", "content":prompt}
      ]
    )

    # Assuming the API returns a dictionary with the response, extract the intent
    # Here we're simulating the structure of a typical OpenAI API response
    # api_response = completion['choices'][0]['message']['content'].strip().split('\n')
    api_response = completion.choices[0].message.content.strip().split('\n')
    return api_response

# def CreateImageGenPrompts(customer_profile, multiple=False):

#   if multiple:
#     prompt = f"""you're an Prompt engineer and now you have been given a  customer profile generate 10 totaly 
#     different yet simple 4-5 words prompt based on the industry, location, profession from this to give to an 
#     marketting image generation model, store just all those 10 prompts in an promot_array it should be strictly a list format and just give me that 
#     prompt_array nothing else, customer profile : {customer_profile}. prompt_array:"""

#   else:
#     prompt = f"""you're an Prompt engineer and now you have been given a  customer profile generate 1 prompt of  
#     10-15 words prompt based on the industry, location, profession from this to give to an marketting image generation 
#     model, store that prompt in an prompt_array it should be strictly a list format and just give me that prompt_array nothing else, 
#     customer profile : {customer_profile}. prompt_array:"""

#   # Get the completion from the OpenAI API
#   completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content":prompt}
#     ]
#   )

#   # Assuming the API returns a dictionary with the response, extract the intent
#   # Here we're simulating the structure of a typical OpenAI API response
#   api_response = completion['choices'][0]['message']['content'].strip().split('\n')
#   print("api responseeeeeeeee:",api_response)
#   response = ast.literal_eval(api_response)
#   print("[1:-1]:       ",response, type(response))

#   return response




def CreateImageGenPrompts(customer_profile, multiple=False):
    print("CreateImageGenPrompts")
    print(multiple)


    if multiple:
        prompt = f"""you're an Prompt engineer and now you have been given a  customer profile generate 10 totaly different 
        yet simple 4-5 words prompt based on the industry, location, profession from this to give to an marketting 
        image generation model, store just all those 10 prompts in an promot_array and just give me that prompt_array 
        nothing else, customer profile : {customer_profile}. prompt_array:"""
        completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content":prompt}
        ]
        )

        # Assuming the API returns a dictionary with the response, extract the intent
        # Here we're simulating the structure of a typical OpenAI API response
        # api_response = completion['choices'][0]['message']['content'].strip().split('\n')
        api_response = completion.choices[0].message.content.strip().split('\n')
        print("multiple response", api_response)
        api_response = ast.literal_eval(api_response[0])
        print(api_response[0], type(api_response[0]))


        return api_response[1:-1]

    else:
        prompt = f"you're an Prompt engineer and now you have been given a  customer profile generate 1 prompt of  10-15 words prompt based on the industry, location, profession from this to give to an marketting image generation model, store ithat prompt in an prompt_array and just give me that prompt_array nothing else, customer profile : {customer_profile}. prompt_array:"
        completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content":prompt}
        ]
        )

        # Assuming the API returns a dictionary with the response, extract the intent
        # Here we're simulating the structure of a typical OpenAI API response
        # api_response = completion['choices'][0]['message']['content'].strip().split('\n')
        api_response = completion.choices[0].message.content.strip().split('\n')
        print("api_response",api_response)

        return api_response
  # Get the completion from the OpenAI API




def GenPictureBank(project_id, ImageGenPrompts=["A professional enviorment"]):


  # generate image 
    image_path = f'/app/app/services/{project_id}.png'

    print("Image_pathhhhhhhhhh", image_path)
    ImageBankByte = []
    for ImagePrompt in ImageGenPrompts:
        print("for loop", ImagePrompt)
        response = together.Image.create(model = "SG161222/Realistic_Vision_V3.0_VAE",
                                        prompt=ImagePrompt)

        # ImageByte = _ImageBankByte[0]...
        print(response)
        ImageBankByte.append(response["output"]["choices"][0])
        print(ImageBankByte)


        image_byte = ImageBankByte[0]
        print(image_byte)
        print("image_byte['image_base64']:    ", image_byte["image_base64"])
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_byte["image_base64"]))
            print(image_path)
        print(image_path)


    return image_path 



def generate_and_save_image(customer_profile, text_color_primary, text_color_secondary,  project_id):
    client = OpenAI()
    image_path = f'/app/app/services/{project_id}.png'
    # Initialize the OpenAI client
    client = OpenAI()
 
    # Construct the prompt
    prompt = f"Design a background for {customer_profile}, using a color scheme that counter contrasts with {text_color_primary} and {text_color_secondary} and themes related to their industry, profession, or location. Ensure the image's brightness counterbalances {text_color_primary} and {text_color_secondary}  for optimal visibility. The image should have no pre-existing text"
 
    # Get the image from the OpenAI API
    print("prompt", prompt)
    completion = client.images.generate(model="dall-e-3",
                                        prompt=prompt,
                                        size="1024x1024",
                                        quality="standard",
                                        n=1)
 
    # Assuming the API returns a dictionary with the response, extract the URL
    image_url = completion.data[0].url
 
    # Download the image
    response = requests.get(image_url)
 
    # Check if the request was successful and save the image
    if response.status_code == 200:
        with open(image_path, 'wb') as file:
            file.write(response.content)
 








 
def asset_creation(table_name, user_id, project_id, extraction_id_brand, extraction_id_tone, file_id_log, file_id_background_image):
    print()
    print("entered")
    process_id = None
    try:
        process_data_insert = [
                    {
                        "user_id" :user_id,
                        "process_type": "asset_creation",
                        "process_status": "in_progress",
                        "start_at" : datetime.now().isoformat()
                    },
                ]


        process= supabase.from_("process").insert(process_data_insert).execute()
        process_data = [record for record in process.data]

        p_data = json.dumps(process_data)
        p = json.loads(p_data)
        process_id = p[0]["process_id"]

        print("process table:*******************", p)





        # table_name = "segment_47b0ffec-356a-4c35-8704-23b153d345c5_1087"

        extracted_data = get_extracted_data(extraction_id_brand, extraction_id_tone)



        primary_color = extracted_data["primary_color"]
        secondary_color = extracted_data["secondary_color"]
        primary_font = extracted_data["primary_font"]
        secondary_font = extracted_data["secondary_font"]
        brand_name = extracted_data["brand_name"]
        tone_of_voice = extracted_data["tone_of_voice"]



        asset_to_individulize_file = fetch_asset_individualize(project_id)




        rgb_colors = get_rgb_colors(primary_color, secondary_color)

        primary_color_rgb = rgb_colors['primary_color_rgb']
        secondary_color_rgb = rgb_colors['secondary_color_rgb']

        logo_path = fetch_logo(file_id_log)

        table = fetch_table_data(table_name)
        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(table)




        # Group the data by the cluster column
        cluster_column = "Cluster"

        grouped_clusters = df.groupby(cluster_column)

        categorical_columns = df.select_dtypes(exclude=[np.number])

        result_filenames = []
        asset_id = []
        asset_path = []

        for cluster_id, cluster_data in grouped_clusters:

            table = "asset_metadata"
            updates = {
              "process_id" : process_id
            }
            response_update = supabase.from_(table).insert(updates).execute()
            
            update_d = [record for record in response_update.data]
            response_u = json.dumps(update_d)
            update= json.loads(response_u)
            update_asset_id = update[0]['id']
            print(update_asset_id)
                    
            # Descriptive statistics for each cluster
            cluster_description = df[df['Cluster'] == cluster_id].describe()
            # print(cluster_description)


            # Descriptive statistics for categorical columns
            categorical_cluster_data = categorical_columns[df['Cluster'] == cluster_id]
            categorical_description = categorical_cluster_data.describe()
        
        #     print("Categorical Column Statistics:")
        #     print(categorical_description)

            # print(f"Cluster Name: {cluster_id} {cluster_description} {categorical_description}")
            


            # if file_type == 'picture_bank':
            if file_id_background_image is not None:

                background_image_path = fetch_background_image(file_id_background_image)

            else:
                # background_image_path = marketing_image_creation(cluster_id, cluster_description, categorical_description, primary_color_rgb, secondary_color_rgb)
                # GenPictureBank
                print("elseeeee")
                cust_profile = CreateCustomerProfile(categorical_description,cluster_description)
                print(cust_profile)
                image_prompts = CreateImageGenPrompts(cust_profile, multiple=True)
                print(image_prompts)
                background_image_path = GenPictureBank(project_id,image_prompts)
                print(background_image_path)

                                # Example usage
                # customer_profile = "YOUR_CUSTOMER_PROFILE"
                # text_color = "YOUR_TEXT_COLOR"
                # image_path = "YOUR_IMAGE_PATH.png"
                
                """background_image_path = generate_and_save_image(cust_profile, primary_color_rgb, secondary_color_rgb, project_id)"""
            
            

            print("asset_to_individulize_file:", asset_to_individulize_file)
            
            asset_text = generate_text(cluster_description, categorical_description, asset_to_individulize_file, tone_of_voice)
            
            print("asset_text", asset_text)



            local_asset_path = combine_text_image(cluster_id, background_image_path, logo_path, asset_text, primary_color_rgb, secondary_color_rgb)
            print("local_asset_path", local_asset_path)
            bucket_path = f"/asset/{user_id}/{project_id}/asset_{cluster_id}.jpg"

                    # print("Bucket Pathhhhhhhhhhhhhhh", bucket_path)
            with open(local_asset_path, 'rb') as f:
                supabase.storage.from_(supabase_bucket).upload(file=f,path=bucket_path)

            print("uploaded image")


    
                
            asset_data_insert = [
                        {
                            "user_id" :user_id,
                            "project_id": project_id,
                            "asset_path": bucket_path
                        },
                    ]


            asset= supabase.from_("asset_metadata").update(asset_data_insert).eq('id',update_asset_id).execute()
            asset_data = [record for record in asset.data]

            p_data = json.dumps(asset_data)
            p = json.loads(p_data)
            print("asssettttttt", p)
            assetid = p[0]["id"]
            print("Asset id---------", assetid)

            asset_id.append(assetid)
            asset_path.append(bucket_path)

            print("process table:*******************", p)
    
    



            process_data_update = {
                            "process_status": "stopped",
                            "end_at" : datetime.now().isoformat()
                        }
            supabase.from_("process").update(process_data_update).eq("process_id", process_id).execute()

            logger.info(f"asset creation done for segment {cluster_id}")

        os.remove(local_asset_path)
        os.remove(background_image_path)
        os.remove(logo_path)
        logger.info("asset creation done")



        return {"asset_id": asset_id, "asset_path": asset_path}
    
    except Exception as e:
        logger.error(e)
        print(e)
        return {"error": e, "status":"error"}



def send_email_with_image(to_email, subject, image_path,filename):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USERNAME
    msg['To'] = to_email
    msg['Subject'] = subject


    url = supabase_client.storage.from_('solarplexus').get_public_url(image_path)

    response = requests.get(url)
    img_file =response.content

    mime_img = MIMEImage(img_file,_subtype="png")
    mime_img.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(mime_img)


    # Connect to the SMTP server and send the email
    try:
        with smtplib.SMTP_SSL(EMAIL_HOST,465) as server:
            server.ehlo()
            server.login(EMAIL_USERNAME,EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, to_email, msg.as_string())
        return f"Email sent to {to_email}"

    except:
        return "Error in sending image to mail"
