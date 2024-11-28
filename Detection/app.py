import cv2
import requests
import json
from huggingface_hub import InferenceClient

# Function to upload an image to Imgur and return the URL
def upload_image_to_imgur(image_path, client_id):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    headers = {
        'Authorization': f'Client-ID {client_id}',
    }

    payload = {
        'image': image_data,
        'type': 'file',
    }

    response = requests.post('https://api.imgur.com/3/upload', headers=headers, files=payload)

    if response.status_code == 200:
        image_url = response.json()['data']['link']
        return image_url
    else:
        return None

# Function to capture an image from the camera and save it locally
def capture_image_from_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' to capture
            image_path = 'captured_image.jpg'
            cv2.imwrite(image_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return image_path

# Function to call Llama API and get description for the image URL
def get_image_description_from_llama(image_url, api_key):
    client = InferenceClient(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Identify the given object. Provide the item name followed by a list of electrical components used in the object. "
                        "List only the component names (e.g., Microcontroller, Button Switch, Optical Sensor) without any descriptions or additional details. "
                        "In addition to identifying components, do the following tasks: \n\n"
                        "1. Suggest possible projects or devices that can be created using the listed components. Describe each project strictly using the identified components only.\n"
                        "2. Create a Serial Flow Diagram for the components, illustrating how they are interconnected and function together.\n"
                        "3. Provide a table listing each component, its pin names, and a brief description of the function for each pin.\n"
                        "5. Calculate the estimated carbon footprint for each component based on standard industry data.\n"
                        "6. Offer a step-by-step reengineering guide using sustainable or eco-friendly alternatives for each component, if available."
                   
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
            messages=messages, 
            max_tokens=2000
        )
        return completion.choices[0].message['content']
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to save all data to JSON format
def save_all_data_to_json(image_url, description, output_file='output.json'):
    # Initialize JSON structure with the image URL
    json_data = {
        "image_url": image_url,
        "description": {
            "item_name": "",
            "components": [],
            "projects": [],
            "serial_flow_diagram": "",
            "components_table": [],
            "project_diagrams": [],
            "carbon_footprint": [],
            "reengineering_guide": ""
        }
    }

    # Now, print the raw description for debugging purposes
    print("Raw description:", description)

    # Split description into relevant sections based on predefined keywords
    sections = description.split('\n\n')

    for section in sections:
        section_lower = section.lower()
        if "item name" in section_lower and ":" in section:
            json_data["description"]["item_name"] = section.split(":", 1)[1].strip()
        elif "components" in section_lower and ":" in section:
            components_list = section.split(":", 1)[1].split(',')
            json_data["description"]["components"] = [component.strip() for component in components_list]
        elif "projects" in section_lower and ":" in section:
            json_data["description"]["projects"] = section.split(":", 1)[1].strip().split('\n')
        elif "serial flow diagram" in section_lower and ":" in section:
            json_data["description"]["serial_flow_diagram"] = section.split(":", 1)[1].strip()
        elif "components table" in section_lower and ":" in section:
            # Assuming a simple table format from the description text
            table_data = section.split(":", 1)[1].strip().split('\n')
            for line in table_data:
                parts = line.split('-')
                if len(parts) == 3:
                    component, pin, description = parts
                    json_data["description"]["components_table"].append({
                        "component": component.strip(),
                        "pin": pin.strip(),
                        "description": description.strip()
                    })
        elif "project diagrams" in section_lower and ":" in section:
            json_data["description"]["project_diagrams"] = section.split(":", 1)[1].strip().split('\n')
        elif "carbon footprint" in section_lower and ":" in section:
            footprint_data = section.split(":", 1)[1].strip().split('\n')
            for line in footprint_data:
                if ':' in line:
                    component, footprint = line.split(':')
                    json_data["description"]["carbon_footprint"].append({
                        "component": component.strip(),
                        "footprint": footprint.strip()
                    })
        elif "reengineering guide" in section_lower and ":" in section:
            json_data["description"]["reengineering_guide"] = section.split(":", 1)[1].strip()

    # Save to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Enhanced description saved to {output_file}")

# Example usage
client_id = ""  # Replace with your Imgur client ID
api_key = ""  # Replace with your Hugging Face API key

# Capture the image from an external camera
image_path = capture_image_from_camera(camera_index=1)  # Adjust camera index if needed

if image_path:
    # Upload the captured image to Imgur
    image_url = upload_image_to_imgur(image_path, client_id)
    if image_url:
        # Get the image description from Llama
        description = get_image_description_from_llama(image_url, api_key)
        
        if description:
            # Save all data to JSON file
            save_all_data_to_json(image_url, description)
        else:
            print("No description received from Llama.")
    else:
        print("Image upload failed.")
