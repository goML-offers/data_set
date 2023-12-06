import os
 
os.environ["OPENAI_API_KEY"] = "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"
 
from openai import OpenAI
import openai 
import requests
client = OpenAI()
# text_summary = "Embrace the speed of innovation with our bespoke solutions for German car manufacturers! At the heart of our approach is AWS's revolutionary IoT and ML technology, turbocharging your production line into the future. Imagine cars not just built, but intelligently crafted with real-time monitoring and predictive maintenance. Our services are the secret ingredient to making your cars not only faster but smarter and more sustainable. Reduced costs, enhanced efficiency, and a greener approach? That's not just progress; it's automotive evolution. Gear up for a journey where every mile is an innovation, tailor-made for the autobahn of tomorrow."


 
# text_color = "(235, 105, 27)"


def image_gen(customer_profile, text_color_primary, text_color_secondary,  project_id):
    image_path = f'/app/app/services/{project_id}.png'

    prompt = f"""Create a realistic banner for a newsletter, size 1792x1024, inspired by the following text 
    summary: '{customer_profile}'. The banner should visually represent the key themes and 
    ideas mentioned in the summary. Include relevant imagery, symbols, or graphics that align 
    with the subject matter. Use a color scheme that complements the tone and content of the 
    summary. Overlay the main title and any important phrases from the summary prominently on 
    the banner. Ensure the design is engaging and appropriate for the target audience of the newsletter."""

    #prompt= "Create an elegant and sophisticated background that resonates with professionals from diverse educational backgrounds, specifically referencing Shiplake College in Henley. Incorporate elements of modern business and education, with subtle nods to technology and industry trends. The color scheme should be cool and muted, with shades of blue, green, and gray, ensuring a striking contrast with the bright orange text color (RGB: 235, 105, 27). The overall feel should be refined and inviting, suitable for a high-level business audience with a mean headcount of around 106 and interests in various industries"
    #prompt = f"Create an image for {customer_profile}, ensuring a background that contrasts sharply with {text_color}. The design must be attractive to [customer profile], featuring themes like [relevant themes for the customer profile], in colors that counter [text color]. The image should be suitable for marketing, visually captivating, and must not include any text"
    #prompt = f"Design a background for {customer_profile}, using a color scheme that counter contrasts with {text_color} and themes related to their industry, profession, or location. Ensure the image's brightness counterbalances {text_color} for optimal visibility. The image should have no pre-existing text"
    # Get the completion from the OpenAI API
    completion = client.images.generate( model="dall-e-3",
                                        prompt = prompt,
                                        size="1024x1024",
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
