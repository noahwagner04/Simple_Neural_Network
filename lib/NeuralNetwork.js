class NeuralNetwork {
	constructor(config) {
		if (config instanceof NeuralNetwork) {
			this.inputDim = config.inputDim;
			this.hiddenDim = config.hiddenDim;
			this.outputDim = config.outputDim;

			this.nodes = Array.from(config.nodes);

			this.layers = [];

			this.weights = [];
			this.biases = [];

			for (let i = 0; i < config.weights.length; i++) {
				this.weights.push(config.weights[i].clone());
				this.biases.push(config.biases[i].clone());
			}

		} else {
			this.inputDim = config.inputNodes;
			this.hiddenDim = config.hiddenNodes;
			this.outputDim = config.outputNodes;

			this.layers = [];

			this.weights = [];
			this.biases = [];
			this.nodes = Array.from(this.hiddenDim);
			this.nodes.push(this.outputDim);
			this.nodes.unshift(this.inputDim);

			for (let i = 0; i < this.nodes.length - 1; i++) {
				this.weights.push(new Matrix(this.nodes[i + 1], this.nodes[i]));
				this.weights[i].randomize();

				this.biases.push(new Matrix(this.nodes[i + 1], 1));
				this.biases[i].randomize();
			}
		}
		this.learningRate = config.learningRate;
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
		this.layers = [];

		let inputs = Matrix.fromArray(inputArray);

		this.layers.push(inputs);

		let feedForward = inputs.clone();

		for (let i = 0; i < this.nodes.length - 1; i++) {
			let hidden = Matrix.multiply(this.weights[i], feedForward);
			hidden.add(this.biases[i]);
			hidden.map(this.activationFunction);
			feedForward = hidden;
			this.layers.push(feedForward);
		}
		return feedForward;
	}

	train(inputArray, targetArray) {
		if (Array.isArray(inputArray) === false || inputArray.length !== this.inputDim) {
			console.log("invalid input");
			return;
		}
		let outputs = this.predict(inputArray);
		let inputs = Matrix.fromArray(inputArray);

		let errors = undefined;

		for (let i = this.layers.length - 1; i >= 1; i--) {
			if (i === this.layers.length - 1) {
				let targets = Matrix.fromArray(targetArray);
				errors = Matrix.subtract(targets, outputs);
			} else {
				let weightsT = Matrix.transpose(this.weights[i]);
				errors = Matrix.multiply(weightsT, errors);
			}
			let deltas = Matrix.map(this.layers[i], this.dActivationFunction);
			deltas.multiply(errors);
			deltas.multiply(this.learningRate);

			let hiddenT = Matrix.transpose(this.layers[i - 1]);
			let weightDeltas = Matrix.multiply(deltas, hiddenT);

			this.weights[i - 1].add(weightDeltas);
			this.biases[i - 1].add(deltas);
		}
	}

	// this returns a deep copy.
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