import fitz  # PyMuPDF
import openai
import json

from dotenv import load_dotenv
import os

load_dotenv()

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

api_key = os.getenv("OPEN_AI_API_KEY")

client = openai.OpenAI(api_key=api_key)

# Example extracted text
pdf_text = extract_text_from_pdf("/Users/mgor/Downloads/case-files/level-2/AHMEDABAD_C-4-2011_27-09-2019.pdf")

# Your expected schema
json_schema = {
    "pdf_file": "string",
    "response": {
        "lowerCourtName": "string",
        "currentCourtName": "string",
        "partyA": "string",
        "partyB": "string",
        "factualBackground": "string",
        "legalIssues": "array_of_strings",
        "arguments": {
            "ArgumentFromPartyA": "string",
            "ArgumentFromPartyB": "string"
        },
        "decisions": "array_of_strings",
        "lowerCourtFavour": "either on of the values partyA or partyB",
        "currentCourtFavour": "either on of the values partyA or partyB",
        "nextPlaceOfAppeal": "string",
        "precedentSearchTerms": "array_of_strings"
    }
}

# Prompt + schema instruction
prompt = f"""
You are an AI assistant. Extract structured data from the following document text and return it in this JSON format:

{json.dumps(json_schema, indent=2)}

Document text:
\"\"\"
{pdf_text[:5000]}  # you can truncate or batch it if needed
\"\"\"
"""

# Send to OpenAI
response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt}
  ],
  temperature=0.2
)

# Parse and print response
print(response)
structured_json = response.choices[0].message.content
print(structured_json)
