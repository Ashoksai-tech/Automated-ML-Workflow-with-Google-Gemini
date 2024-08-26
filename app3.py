import google.generativeai as genai
import os

genai.configure(api_key=os.environ["AIzaSyDNMUxIgTCct7U4aL4Utyup_sfEr7aZI7I"])

model = genai.GenerativeModel('gemini-1.5-flash')

response = model.generate_content("Write a story about an AI and magic")
print(response.text)