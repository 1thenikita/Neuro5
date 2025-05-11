from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
import uvicorn
from neural_network import Neuron, SimpleNeuralNetwork

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def load_network(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

network = load_network('trained_network.pkl')
@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("guns.html", {"request": request})

@app.post("/classify", response_class=HTMLResponse)
async def classify_vegetable(request: Request, length: float = Form(...), width: float = Form(...)):
    try:
        inputs = np.array([[length, width]])
        output = network.feedforward(inputs[0])
        predicted_class = np.argmax(output)
        vegetable_classes = {0: "Винтовка", 1: "Граната", 2: "Пистолет"}
        result = vegetable_classes[predicted_class]
        return templates.TemplateResponse("guns_result.html", {
            "request": request,
            "length": length,
            "width": width,
            "result": result,
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app)