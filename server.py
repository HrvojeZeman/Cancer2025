import \
	numpy
from fastapi import FastAPI, File, UploadFile
from NN import model
import os

app = FastAPI()

UPLOAD_DIR = r"C:\Users\Hrvoje\OneDrive - Univerza v Mariboru\Namizje\Library of Congress\Moji prispevki\AES Enkripcija\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def process(file: UploadFile = File(...)):
	NumpyMatrika = numpy.array(UploadFile)
	
	return model.predict(NumpyMatrika)