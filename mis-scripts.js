// Variables globales para los modelos
let faceDetectionModel = null;
let emotionModel = null;

// Inicializar los modelos
async function initializeModels() {
	try {
		// Cargar el modelo de detección facial
		faceDetectionModel = await blazeface.load();
		// Cargar el modelo de emociones
		emotionModel = await tf.loadLayersModel('./tensorflowjs-model/model.json');
		console.log("Modelos cargados exitosamente");
	} catch (error) {
		console.error("Error al cargar los modelos:", error);
	}
}

// Inicializar el modelo de detección facial
async function initializeFaceDetection() {
	if (!faceDetectionModel) {
		faceDetectionModel = await blazeface.load();
	}
}

// Detectar el rostro más grande
async function findBiggestFace(imageElement) {
	try {
		const predictions = await faceDetectionModel.estimateFaces(imageElement, false);
		if (predictions.length === 0) {
			return null;
		}

		// Encontrar el rostro más grande
		let biggestFace = predictions[0];
		let maxArea = 0;

		for (const face of predictions) {
			const width = face.bottomRight[0] - face.topLeft[0];
			const height = face.bottomRight[1] - face.topLeft[1];
			const area = width * height;
			
			if (area > maxArea) {
				maxArea = area;
				biggestFace = face;
			}
		}

		return biggestFace;
	} catch (error) {
		console.error("Error en la detección facial:", error);
		return null;
	}
}

// Clasificar la emoción de la imagen mostrada
async function predict_emotion() {
	let input = document.getElementById("input_image");
	const start = performance.now();
	
	// Detectar el rostro más grande
	const face = await findBiggestFace(input);
	
	// Si no se detecta un rostro, usar la imagen completa
	let imageproc;
	const croppedContainer = document.getElementById("container-cropped");
	const croppedImage = document.getElementById("cropped_image");
	
	if (face) {
		// Crear un canvas para el rostro
		const canvas = document.createElement('canvas');
		canvas.width = 299;
		canvas.height = 299;
		const ctx = canvas.getContext('2d');
		
		// Recortar el rostro
		ctx.drawImage(
			input,
			face.topLeft[0],
			face.topLeft[1],
			face.bottomRight[0] - face.topLeft[0],
			face.bottomRight[1] - face.topLeft[1],
			0, 0, 299, 299
		);
		
		// Mostrar la imagen recortada
		croppedImage.src = canvas.toDataURL('image/png');
		croppedContainer.style.display = 'block';
		
		imageproc = tf.browser.fromPixels(canvas)
			.resizeNearestNeighbor([299, 299])
			.toFloat()
			.div(127.5)
			.sub(1.0)
			.expandDims(0);
	} else {
		// Ocultar el contenedor de imagen recortada
		croppedContainer.style.display = 'none';
		
		// Usar la imagen completa si no se detecta rostro
		imageproc = tf.browser.fromPixels(input)
			.resizeNearestNeighbor([299, 299])
			.toFloat()
			.div(127.5)
			.sub(1.0)
			.expandDims(0);
	}
	
	console.log("Finalización del preprocesamiento de la imagen");

	const pred = emotionModel.predict(imageproc);
	pred.print();
	console.log("Finalización de predicción");

	const emotions = ["Enojado", "Asustado", "Feliz", "Neutral", "Triste"];

	pred.data().then((data) => {
		console.log("Prediction data:", data);

		const output = document.getElementById("output_text");
		output.innerHTML = "";

		// Encontrar el índice del valor más alto (primera emoción)
		let max_val = -1, max_val_index = -1;
		let second_val = -1, second_val_index = -1;

		for (let i = 0; i < data.length; i++) {
			if (data[i] > max_val) {
				second_val = max_val;
				second_val_index = max_val_index;
				max_val = data[i];
				max_val_index = i;
			} else if (data[i] > second_val) {
				second_val = data[i];
				second_val_index = i;
			}
		}

		const confidence = max_val * 100;
		const second_confidence = second_val * 100;
		const end = performance.now();
		const processingTime = ((end - start) / 1000).toFixed(2);

		if (confidence < 50) {
			output.innerHTML = `
				<p>No se reconoce la emoción en la imagen.</p>
				<p>Confianza máxima: ${confidence.toFixed(2)}%</p>
				<p>Tiempo de procesamiento: ${processingTime} s</p>
			`;
		} else {
			const EMOCION_DETECTADA = emotions[max_val_index];
			const SEGUNDA_EMOCION = emotions[second_val_index];
			output.innerHTML = `
				<p>La emoción detectada es:</p>
				<p><strong>${EMOCION_DETECTADA}</strong> (${confidence.toFixed(2)}% probabilidad)</p>
				<p>Segunda opción: <strong>${SEGUNDA_EMOCION}</strong> (${second_confidence.toFixed(2)}% probabilidad)</p>
				<p>Tiempo de procesamiento: ${processingTime} s</p>
			`;
		}
	});
}

// Acceder a la cámara y capturar una imagen después de 5 segundos
window.onload = async function () {
	const video = document.getElementById('video');
	const img = document.getElementById('input_image');
	const output = document.getElementById('output_text');
	const countdownElem = document.getElementById('countdown');

	// Inicializar los modelos
	await initializeModels();

	// Inicializar el modelo de detección facial
	await initializeFaceDetection();

	try {
		const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 299, height: 299 } });
		video.srcObject = stream;
		video.style.display = 'block';
		img.style.display = 'none';

		let seconds = 5;
		countdownElem.textContent = `Foto en ${seconds}...`;
		const interval = setInterval(() => {
			seconds--;
			if (seconds > 0) {
				countdownElem.textContent = `Foto en ${seconds}...`;
			} else {
				clearInterval(interval);
				countdownElem.textContent = '';
				// Tomar la foto
				const canvas = document.createElement('canvas');
				canvas.width = 299;
				canvas.height = 299;
				const ctx = canvas.getContext('2d');
				ctx.drawImage(video, 0, 0, 299, 299);

				stream.getTracks().forEach(track => track.stop());

				img.src = canvas.toDataURL('image/png');
				img.style.display = 'block';
				video.style.display = 'none';

				img.onload = () => {
					predict_emotion();
				};
			}
		}, 1000);
	} catch (err) {
		output.innerHTML = "No se pudo acceder a la cámara.";
		console.error(err);
	}
};

// Desplegar imagen subida por el usuario
function display(event) {
	let input_image = document.getElementById("input_image");
	input_image.src = URL.createObjectURL(event.target.files[0]);
	
	console.log(input_image.src);
	let d = document.querySelector(".path");
	d.textContent = "La URL de esta imagen es: " + input_image.src;
	
	// Inicializar el modelo de detección facial si no está cargado
	if (!faceDetectionModel) {
		initializeFaceDetection().then(() => {
			predict_emotion();
		});
	} else {
		predict_emotion();
	}
}
