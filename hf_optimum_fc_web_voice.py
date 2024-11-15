# Demonstrate running on intel (no Nvidia)
# Optimize the models with optimum to run on openvino
# POC of function calling, speech2text
# There is a corresponding version running on Nvidia
# pip install optimum[openvino,nncf]

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
import re
import json
import requests
import yfinance as yf
import pyaudio
import wave
import threading
# Install numpy == 1.26.4. do not upgrade. higher version does not work with some of the libraries.

# Here we initialize the FastAPI application and set up Jinja2 for template rendering
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="/users/chiauho.ong/LLM/hf_optimum_function_calling/static"), name="static")

#################### ALL THE AUDIO STUFF ######################################################################################
# This section of the code deals with audio / speech input.The users will use a mic to speak instead of typing.
# This python code process the HTML UI request. The code records the speech and save it into a file "voice.wav".
# Audio recording parameters
import librosa
from transformers import AutoProcessor
from optimum.intel import OVModelForSpeechSeq2Seq

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
LANGUAGE = "en"
WAVE_OUTPUT_FILENAME = "voice.wav"

audio = pyaudio.PyAudio()
frames = []
recording = False
record_thread = None

# Load the Whisper model and processor for transcribing
model_id = "openai/whisper-large-v3"
ov_model_id = "C:/Users/chiauho.ong/.cache/huggingface/hub/hf_optimum-whisper-large-v3"    # my convert model using optimum

print("Load tokenizer")
tokenizer = AutoProcessor.from_pretrained(model_id)
print("Load model")
model = OVModelForSpeechSeq2Seq.from_pretrained(ov_model_id)


class LLMRequest(BaseModel):
    transcription: str


def record_audio():
    global recording, frames
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
# end of function


@app.post("/start_recording")
async def start_recording():
    global recording, frames, record_thread
    if not recording:
        recording = True
        frames = []
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()
        return {"status": "success", "message": "Recording started"}
    raise HTTPException(status_code=400, detail="Already recording")


@app.post("/stop_recording")
async def stop_recording():
    global recording, record_thread
    if recording:
        recording = False
        if record_thread:
            record_thread.join()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        try:
            # Load the audio file
            speech, samplerate = librosa.load("voice.wav", sr=16000)
            raw_speech = speech.tolist()
            input_ids = tokenizer(raw_speech, sampling_rate=RATE, return_tensors="pt")
            outputs = model.generate(**input_ids, language=LANGUAGE)
            transcription = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Transription is: {transcription}")
            r = JSONResponse(content={"output_text": transcription.strip()})
            return r
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")


@app.post("/process_llm")
async def process_llm(request: LLMRequest):
    try:
        transcription = request.transcription  # Access the transcription field from the request

        # Send transcription to LLM and get the result
        s, i = submit_query(transcription)
        s = transcription + "\n\n" + s
        i = [i]
        return JSONResponse(content={"output_text": s.strip(), "images": i})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM processing: {str(e)}")


################################### END ALL THE AUDIO STUFF ###########################################################

# This defines a Pydantic model for validating user input.
class UserInput(BaseModel):
    user_input: str


# This route handler serves the main HTML page.
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_input")
async def process_input(user_input: UserInput):
    s, i = submit_query(user_input.user_input)
    i = [i]
    r = JSONResponse(content={"output_text": s, "images": i})
    return r


@app.post("/get_images")
async def initial_image():
    i = "/static/coffee_machine.jpg"
    i = [i]
    r = JSONResponse(content={"output_text": "A coffee machine", "images": i})
    return r


@app.get("/get_images")
async def get_images():
    i = "/static/vending_machine_v2.png"
    i = [i]
    r = JSONResponse(content={"output_text": "A coffee machine", "images": i})
    return r


def prompt_formatter(u_query, system_prompt, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": u_query}
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt
# End of function


def llm(u_query, system_prompt):
    global g_model, g_tokenizer

    print(f"Format prompt ... \n")
    prompt = prompt_formatter(u_query, system_prompt, g_tokenizer)
    print(f"Prompt: {prompt}\n")

    inputs = g_tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Start llm\n")
    outputs = g_model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.01)
    output_text = g_tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print("End llm\n")
    print(output_text)  # Expected output: <tool_call>{{"tool_name": "<function-name>", "tool_arguments": <args-dict>}}</tool_call>

    function = extract_function_from_xml(output_text)
    print(f"Function to call is {function}")

    return function     # Function name in str
# End of function


def submit_query(user_input: str):
    global g_system_prompt
    print("User Query:", user_input)  # Stored user input passed from web UI

    print(f"Call llm ... \n")
    function = llm(user_input, g_system_prompt)

    # Prepare the exec string
    exec_locals = {}    # local dict to store any local variables that exec() modifies
    r = "s, i = " + function

    # The exec() function is called with globals() and exec_locals as arguments.
    # This ensures that any local variables created inside exec() are stored in the exec_locals dictionary.
    exec(r, globals(), exec_locals)

    # Extract the values of s and i from exec_locals
    s = exec_locals.get('s')
    i = exec_locals.get('i')
    return s, i
# end function


#######################################################################################################################################################
def get_stock_price(symbol):
    # Fetch the stock data
    stock = yf.Ticker(symbol)

    # Check if 'open' exists in the stock info dictionary
    if 'open' in stock.info:
        current_price = stock.info['open']
        s = f"Open price for {symbol} is " + str(current_price)
        image = "/static/stock_price.jpg"
    else:
        # Handle the case where the key does not exist
        current_price = None
        s = f"Could not retrieve the open price for {symbol}. Key 'open' not found."
        image = "/static/error.jpg"
    return s, image
# end of function


def get_weather(api_key="d2f1bddd18b7df4ded8003d6132f6b80", city="Singapore"):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    # Construct the full API URL
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"

    # Send a GET request to the API
    response = requests.get(complete_url)

    # Parse the response in JSON format
    weather_data = response.json()

    # Check if the response was successful
    if weather_data["cod"] != "404":
        main = weather_data["main"]
        weather = weather_data["weather"][0]

        # Extract relevant information
        temperature = main["temp"]
        pressure = main["pressure"]
        humidity = main["humidity"]
        weather_description = weather["description"]

        # Display the results
        s = f"The weather in {city} is as follows:\nTemperature: {temperature}°C\nPressure: {pressure} hPa\nHumidity: {humidity}%\nDescription: {weather_description.capitalize()}"
        image = "/static/weather.jpg"
    else:
        s = "City not found. Please check the city name."
        image = "/static/error.jpg"
    return s, image
# end of function


def make_coffee(types_of_coffee='long black', milk='normal', sugar='normal', strength='normal'):
    s = f"""Making a cup of {types_of_coffee} with the following options:
Milk = {milk},
Sugar = {sugar}
Strength = {strength}
"""
    image_case = {
        "cappuccino": "/static/cappuccino.jpg",
        "latte": "/static/latte.jpg",
        "americano": "/static/americano.jpg",
        "long black": "/static/long_black.jpg"
    }
    image = image_case.get(types_of_coffee, "/static/black_coffee.jpg")
    return s, image
# end of function


def cook_burger(cook="well done"):
    s = f"Cooking a beef burger that is {cook}"
    image = "/static/beef_burger.jpg"
    return s, image
# end of function


def cook_fries(type_of_fries="straight cut"):
    s = f"Cooking {type_of_fries} fries"
    image_case = {
        "straight cut": "/static/straight_cut_fries.jpg",
        "curly": "/static/curly_fries.jpg",
    }
    image = image_case.get(type_of_fries, "/static/straight_cut_fries.jpg")
    return s, image
# end of function


def cook_prawn_noodles(prawn="with prawn", sotong="with sotong"):
    s = f"""Cooking fried prawn noodles with the following options:
Prawn = {prawn},
Sotong = {sotong}
"""
    image = "/static/prawn_noodle.jpg"
    return s, image
# end of function


def extract_function_from_xml(xml_string):
    # Extract JSON from the tool_call string
    json_match = re.search(r'{.*}', xml_string)
    if json_match:
        json_str = json_match.group(0)

        # Parse the JSON string
        data = json.loads(json_str)

        # Extract function name and parameters
        tool_name = data.get('tool_name')
        tool_arguments = data.get('tool_arguments', {})

        # Create function call string with actual parameter values
        arguments = ', '.join([f"{key}='{value}'" for key, value in tool_arguments.items()])
        return f"{tool_name}({arguments})" if arguments else f"{tool_name}()"

    return None
# end of function


def create_system_prompt(tools_list):
    system_prompt_format = """You are a function calling AI model. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into function. The user may use the terms function calling or tool use interchangeably.

Here are the available functions:
<tools>{}</tools>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags in the format:
<tool_call>{{"tool_name": "<function-name>", "tool_arguments": <args-dict>}}</tool_call>"""

    # Convert the tools list to a string representation with proper formatting
    tools_str = "\n".join([f"<tool>{tool}</tool>" for tool in tools_list])

    # Format the system prompt with the tools list
    system_prompt = system_prompt_format.format(tools_str)

    return system_prompt
# end of function


if __name__ == "__main__":
    import uvicorn

    global g_tokenizer, g_system_prompt
    global g_model

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    OV_model_id = "C:/Users/chiauho.ong/.cache/huggingface/hub/hf_optimum_auto-Llama-3.2-3B-Instruct"
    g_tokenizer = AutoTokenizer.from_pretrained(model_id)
    g_model = OVModelForCausalLM.from_pretrained(OV_model_id)

    tools_list = [
        {
            "name": "cook_burger",
            "description": "beef burger",
            "parameters": {
                "cook": {
                    "description": "Can be well done, medium or rare",
                    "type": "str",
                    "default": "well done"
                }
            }
        },
        {
            "name": "cook_fries",
            "description": "Potatoes fries",
            "parameters": {
                "type_of_fries": {
                    "description": "Can be straight cut or curly",
                    "type": "str",
                    "default": "straight cut"
                }
            }
        },
        {
            "name": "cook_prawn_noodles",
            "description": "Fried prawn noodles",
            "parameters": {
                "prawn": {
                    "description": "Options for prawn. Can be with or without prawn.",
                    "type": "str",
                    "default": "with prawn"
                },
                "sotong": {
                    "description": "Options for sotong. Can be with or without sotong.",
                    "type": "str",
                    "default": "with sotong"
                }
            }
        },
        {
            "name": "make_coffee",
            "description": "Customer orders coffee.",
            "parameters": {
                "types_of_coffee": {
                    "description": "The type of coffee. Examples are latte, americano, cappuccino. It could also be just coffee or black coffee.",
                    "type": "str",
                    "default": "long black"
                },
                "milk": {
                    "description": "Options for milk with the coffee. Can be 'normal', 'no', 'more', 'less'.",
                    "type": "str",
                    "default": "normal"
                },
                "sugar": {
                    "description": "Options for sugar with the coffee. Can be 'normal', 'no', 'more', 'less'.",
                    "type": "str",
                    "default": "normal"
                },
                "strength": {
                    "description": "Options for coffee strength. Can be 'normal', 'strong', 'weak'.",
                    "type": "str",
                    "default": "normal"
                }
            }
        },
        {
            "name": "get_stock_price",
            "description": "Retrieves the current stock price given a stock symbol.",
            "parameters": {
                "symbol": {
                    "description": "The stock symbol for which the price is what we wanted. Example of stock symbol is HPQ.",
                    "type": "str",
                    "default": ""
                }
            }
        },
        {
            "name": "get_weather",
            "description": "A function that retrieves the current weather for a given city.",
            "parameters": {
                "city": {
                    "description": "The city which we want to know the weather  (e.g., 'New York' or 'Singapore').",
                    "type": "str",
                    "default": "Tokyo"
                }
            }
        }
    ]

    # Create the system prompt with the tools list
    g_system_prompt = create_system_prompt(tools_list)

    # Run Fast API
    uvicorn.run(app, host="0.0.0.0", port=80)
