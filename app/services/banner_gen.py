import os
 


from openai import OpenAI
import openai 
import requests
import together
import base64

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

client = OpenAI()
# text_summary = "Embrace the speed of innovation with our bespoke solutions for German car manufacturers! At the heart of our approach is AWS's revolutionary IoT and ML technology, turbocharging your production line into the future. Imagine cars not just built, but intelligently crafted with real-time monitoring and predictive maintenance. Our services are the secret ingredient to making your cars not only faster but smarter and more sustainable. Reduced costs, enhanced efficiency, and a greener approach? That's not just progress; it's automotive evolution. Gear up for a journey where every mile is an innovation, tailor-made for the autobahn of tomorrow."


 
# text_color = "(235, 105, 27)"


# def image_gen(industry, text_color_primary, text_color_secondary,  project_id):
#     image_path = f'/app/app/services/{project_id}.png'

#     prompt = f"""Create a glossy artistic yet neat and simple designed banner for the {industry} industry newsletter, dont write any text on the image only visual"""

#     #prompt= "Create an elegant and sophisticated background that resonates with professionals from diverse educational backgrounds, specifically referencing Shiplake College in Henley. Incorporate elements of modern business and education, with subtle nods to technology and industry trends. The color scheme should be cool and muted, with shades of blue, green, and gray, ensuring a striking contrast with the bright orange text color (RGB: 235, 105, 27). The overall feel should be refined and inviting, suitable for a high-level business audience with a mean headcount of around 106 and interests in various industries"
#     #prompt = f"Create an image for {customer_profile}, ensuring a background that contrasts sharply with {text_color}. The design must be attractive to [customer profile], featuring themes like [relevant themes for the customer profile], in colors that counter [text color]. The image should be suitable for marketing, visually captivating, and must not include any text"
#     #prompt = f"Design a background for {customer_profile}, using a color scheme that counter contrasts with {text_color} and themes related to their industry, profession, or location. Ensure the image's brightness counterbalances {text_color} for optimal visibility. The image should have no pre-existing text"
#     # Get the completion from the OpenAI API
#     completion = client.images.generate( model="dall-e-3",
#                                         prompt = prompt,
#                                         size="1024x1024",
#                                         quality="standard",
#                                         n=1)
    
#     # Assuming the API returns a dictionary with the response, extract the intent
#     # Here we're simulating the structure of a typical OpenAI API response
#     api_response = completion.data[0].url
    
#     print(api_response)
#     image_url = api_response
#     response = requests.get(image_url)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Open a file in binary write mode
#         with open(image_path, 'wb') as file:
#             file.write(response.content)

#     return image_path


def image_gen(industry, text_color_primary, text_color_secondary,  project_id,headline):
    image_path = f'/app/app/services/{project_id}.png'

    # prompt = f"""Create a banner for the {industry} industry newsletter, newsletter : {newsletter_content}. dont write any text on the image only visual"""

    prompt = f"""Create a glossy banner with a professional background and only visuals for the {industry} industry newsletter, newsletter : {headline}"""


    #prompt= "Create an elegant and sophisticated background that resonates with professionals from diverse educational backgrounds, specifically referencing Shiplake College in Henley. Incorporate elements of modern business and education, with subtle nods to technology and industry trends. The color scheme should be cool and muted, with shades of blue, green, and gray, ensuring a striking contrast with the bright orange text color (RGB: 235, 105, 27). The overall feel should be refined and inviting, suitable for a high-level business audience with a mean headcount of around 106 and interests in various industries"
    #prompt = f"Create an image for {customer_profile}, ensuring a background that contrasts sharply with {text_color}. The design must be attractive to [customer profile], featuring themes like [relevant themes for the customer profile], in colors that counter [text color]. The image should be suitable for marketing, visually captivating, and must not include any text"
    #prompt = f"Design a background for {customer_profile}, using a color scheme that counter contrasts with {text_color} and themes related to their industry, profession, or location. Ensure the image's brightness counterbalances {text_color} for optimal visibility. The image should have no pre-existing text"
    # Get the completion from the OpenAI API
    completion = client.images.generate( model="dall-e-3",
                                        prompt = prompt,
                                        size="1792x1024",
                                        quality="standard",
                                        n=1)
    
    # Assuming the API returns a dictionary with the response, extract the intent
    # Here we're simulating the structure of a typical OpenAI API response
    api_response = completion.data[0].url
    
    print(api_response)
    image_url = api_response
    response = requests.get(image_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode
        with open(image_path, 'wb') as file:
            file.write(response.content)

    return image_path


def GenBgImage(project_id, industry):


  # generate image 
    image_path = f'/app/app/services/{project_id}.png'

    print("Image_pathhhhhhhhhh", image_path)


    #print("Prompt : ", ImagePrompt)
    response = together.Image.create(model = "SG161222/Realistic_Vision_V3.0_VAE",
                                    prompt=f"realistic {industry} background", width=1792, height=1024)

    # ImageByte = _ImageBankByte[0]...
    print(response)
    image_byte = response["output"]["choices"][0]
   



    print(image_byte)
    print("image_byte['image_base64']:    ", image_byte["image_base64"])
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_byte["image_base64"]))
        print(image_path)
    print(image_path)


    return image_path 
