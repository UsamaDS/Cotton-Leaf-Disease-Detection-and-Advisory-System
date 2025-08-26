Cotton Leaf Disease Detection and Bi-Lingual Advisory System (YOLO + LLM + Web UI)

* This project is part of GIKI AI Bootcamp.
* In order to run the code download Anaconda and clone the environment with AnacondaEnvironment.yaml
* If you want to use LLM then you have to specify the OpenAI/Grok API key
  * To use OpenAI LLM in the advisory.py file set the variable: api_key at line 170 with your OpenAI API_KEY and in the function definition change the llmChoice parameter with "openai" instead of "grok"
  * To use Grok LLM in the advisory.py file set the variable: groq_api_key at line 212 with your Grok API_KEY and in the function definition change the llmChoice parameter with "grok"

After Settting up your parameters:
  * open terminal and enter following command to start stremlit service: "streamlit run app.py" Make sure all of your files are in the folder pointed by your terminal
  * After running above command in your browser open this address: http://localhost:8501 and enjoy the project.
