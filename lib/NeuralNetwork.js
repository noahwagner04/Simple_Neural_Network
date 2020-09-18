class NeuralNetwork {
	constructor(config) {
		if (config instanceof NeuralNetwork) {
			this.inputDim = config.inputDim;
			this.hiddenDim = config.hiddenDim;
			this.outputDim = config.outputDim;

			this.weightsIH = config.weightsIH.clone();
			this.weightsHO = config.weightsHO.clone();

			this.biasH = config.biasH.clone();
			this.biasO = config.biasO.clone();

			this.learningRate = config.learningRate;

		} else {
			this.inputDim = config.inputNodes;
			this.hiddenDim = config.hiddenNodes;
			this.outputDim = config.outputNodes;

			this.weightsIH = new Matrix(this.hiddenDim, this.inputDim);
			this.weightsHO = new Matrix(this.outputDim, this.hiddenDim);

			this.weightsIH.randomize();
			this.weightsHO.randomize();

			this.biasH = new Matrix(this.hiddenDim, 1);
			this.biasO = new Matrix(this.outputDim, 1);

			this.biasH.randomize();
			this.biasO.randomize();

			this.learningRate = config.learningRate;
		}
	}

	activationFunction(x) {
		return 1 / (1 + Math.exp(-x));
	}

	dActivationFunction(x) {
		return x * (1 - x);
	}

	predict(inputArray) {
		if (Array.isArray(inputArray) === false || inputArray.length !== this.inputDim) {
			console.log("invalid input");
			return;
		}
		let inputs = Matrix.fromArray(inputArray);
		let hidden = Matrix.multiply(this.weightsIH, inputs);
		hidden.add(this.biasH);
		hidden.map(this.activationFunction);

		let outputs = Matrix.multiply(this.weightsHO, hidden);
		outputs.add(this.biasO);
		outputs.map(this.activationFunction);

		return outputs;
	}

	train(inputArray, targetArray) {
		if (Array.isArray(inputArray) === false || inputArray.length !== this.inputDim) {
			console.log("invalid input");
			return;
		}
		let inputs = Matrix.fromArray(inputArray);
		let hidden = Matrix.multiply(this.weightsIH, inputs);
		hidden.add(this.biasH);
		hidden.map(this.activationFunction);

		let outputs = Matrix.multiply(this.weightsHO, hidden);
		outputs.add(this.biasO);
		outputs.map(this.activationFunction);

		let targets = Matrix.fromArray(targetArray);

		let outputErrors = Matrix.subtract(targets, outputs);

		let deltasHO = Matrix.map(outputs, this.dActivationFunction);
		deltasHO.multiply(outputErrors);
		deltasHO.multiply(this.learningRate);

		let hiddenT = Matrix.transpose(hidden);
		let weightHODeltas = Matrix.multiply(deltasHO, hiddenT);

		this.weightsHO.add(weightHODeltas);
		this.biasO.add(deltasHO);

		let weightsHOT = Matrix.transpose(this.weightsHO);
		let hiddenErrors = Matrix.multiply(weightsHOT, outputErrors);

		let deltasIH = Matrix.map(hidden, this.dActivationFunction);
		deltasIH.multiply(hiddenErrors);
		deltasIH.multiply(this.learningRate);

		let inputT = Matrix.transpose(inputs);
		let weightIHDeltas = Matrix.multiply(deltasIH, inputT);

		this.weightsIH.add(weightIHDeltas);
		this.biasH.add(deltasIH);
	}

	clone() {
		return new NeuralNetwork(this);
	}

	mutate(func) {
		this.weightsIH.map(func);
		this.weightsHO.map(func);
		this.biasH.map(func);
		this.biasO.map(func);
	}
}