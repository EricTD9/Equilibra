// Desplegar imagen subida por el usuario
function display(event) {
	let input_image = document.getElementById("input_image");
	input_image.src = URL.createObjectURL(event.target.files[0]);
	
	console.log(input_image.src);
	let d = document.querySelector(".path");
	d.textContent = "La URL de esta imagen es: " + input_image.src;
}

// Acceder a la cámara y capturar una imagen después de 5 segundos
window.onload = async function () {
	const video = document.getElementById('video');
	const img = document.getElementById('input_image');
	const output = document.getElementById('output_text');

	// Solicitar acceso a la cámara
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 299, height: 299 } });
		video.srcObject = stream;
		video.style.display = 'block';
		img.style.display = 'none';

		// Esperar 5 segundos y capturar la imagen
		setTimeout(() => {
			// Crear un canvas para capturar el frame
			const canvas = document.createElement('canvas');
			canvas.width = 299;
			canvas.height = 299;
			const ctx = canvas.getContext('2d');
			ctx.drawImage(video, 0, 0, 299, 299);

			// Detener la cámara
			stream.getTracks().forEach(track => track.stop());

			// Mostrar la imagen capturada
			img.src = canvas.toDataURL('image/png');
			img.style.display = 'block';
			video.style.display = 'none';

			// Realizar la predicción de emoción
			predict_emotion();
		}, 5000);
	} catch (err) {
		output.innerHTML = "No se pudo acceder a la cámara.";
		console.error(err);
	}
};

// Clasificar la emoción de la imagen mostrada
async function predict_emotion() {
	let input = document.getElementById("input_image");
	
	const start = performance.now();
	
	let imageproc = tf.browser.fromPixels(input)
    .resizeNearestNeighbor([299, 299]) // Cambia el tamaño si tu modelo lo requiere
    .toFloat()
    .div(127.5)
    .sub(1.0)
    .expandDims(0);
	console.log("Finalización del preprocesamiento de la imagen");

	const model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
	const pred = model.predict(imageproc);
	pred.print();
	console.log("Finalización de predicción");

	const emotions = ["Enojado", "Asustado", "Feliz", "Neutral", "Triste"];

	pred.data().then((data) => {
		console.log(data);
		console.log("Prediction data:", data);

		const output = document.getElementById("output_text");
		output.innerHTML = "";

		let max_val = -1;
		let max_val_index = -1;

		for (let i = 0; i < data.length; i++) {
			if (data[i] > max_val) {
				max_val = data[i];
				max_val_index = i;
			}
		}

		const confidence = max_val * 100;
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
			output.innerHTML = `
				<p>La emoción detectada es:</p>
				<p><strong>${EMOCION_DETECTADA}</strong> (${confidence.toFixed(2)}% probabilidad)</p>
				<p>Tiempo de procesamiento: ${processingTime} s</p>
			`;
		}
	});
}
