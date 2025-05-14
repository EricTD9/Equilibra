// Desplegar imagen subida por el usuario
function display(event) {
	let input_image = document.getElementById("input_image");
	input_image.src = URL.createObjectURL(event.target.files[0]);

	console.log(input_image.src);
	let d = document.querySelector(".path");
	d.textContent = "La URL de esta imagen es: " + input_image.src;
}

// Mostrar a qué animal (clase) pertenece la imagen subida
async function predict_animal() {
	let input = document.getElementById("input_image");

	// Inicia medición de tiempo
	const start = performance.now();

	let imageproc = tf.browser.fromPixels(input)
		.resizeNearestNeighbor([224, 224])  // InceptionV3 espera 299x299
		.toFloat()
		.div(127.5)
		.sub(1.0)
		.expandDims(0);
	console.log("Finalización del preprocesamiento de la imagen");

	const model = await tf.loadGraphModel('./tensorflowjs-model_tfjs/model.json');
	const pred = model.predict(imageproc);
	pred.print();
	console.log("Finalización de predicción");

	const animals = ["Enojado", "Asustado", "Feliz", "Neutral", "Triste"];

	pred.data().then((data) => {
		console.log(data);

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
		const end = performance.now(); // Fin de medición de tiempo
		const processingTime = ((end - start) / 1000).toFixed(2); // En segundos

		if (confidence < 50) {
			output.innerHTML = `
				<p>No se reconoce la imagen como ninguna de las clases conocidas.</p>
				<p>Confianza máxima: ${confidence.toFixed(2)}%</p>
				<p>Tiempo de procesamiento: ${processingTime} s</p>
			`;
		} else {
			const ANIMAL_DETECTADO = animals[max_val_index];
			output.innerHTML = `
				<p>El animal detectado es:</p>
				<p><strong>${ANIMAL_DETECTADO}</strong> (${confidence.toFixed(2)}% probabilidad)</p>
				<p>Tiempo de procesamiento: ${processingTime} s</p>
			`;
		}
	});
}
