class Matrix {
	constructor(rows, cols) {
		this.rows = rows;
		this.cols = cols;
		this.data = Array.from(new Array(this.rows), () => new Array(this.cols))
	}

	add(value) {
		if (value instanceof Matrix) {
			if (value.rows !== this.rows || value.cols !== this.cols) {
				console.log("invalid matrix");
				return;
			}
			this.map((e, i, j) => e + value.data[i][j]);
		} else if(!isNaN(value)){
			this.map(e => e + value);
		}
		return this;
	}

	subtract(value) {
		if (value instanceof Matrix) {
			if (value.rows !== this.rows || value.cols !== this.cols) {
				console.log("invalid matrix");
				return;
			}
			this.map((e, i, j) => e - value.data[i][j]);
		} else if(!isNaN(value)){
			this.map(e => e - value);
		}
		return this;
	}

	multiply(value) {
		if (value instanceof Matrix) {
			if (value.rows !== this.rows || value.cols !== this.cols) {
				console.log("invalid matrix");
				return;
			}
			this.map((e, i, j) => e * value.data[i][j]);
		} else if(!isNaN(value)){
			this.map(e => e * value);
		}
		return this;
	}

	static multiply(a, b) {
		if (a instanceof Matrix && b instanceof Matrix) {
			if(a.cols !== b.rows) {
				console.log("invalid matrixes");
				return;
			} 
			return new Matrix(a.rows, b.cols)
				.map((e, i, j) => {
					let result = 0;
					for (let k = 0; k < a.cols; k++) {
						result += a.data[i][k] * b.data[k][j];
					}
					return result;
				});
		}
	}

	map(func) {
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				let value = this.data[i][j];
				this.data[i][j] = func(value, i, j);
			}
		}
		return this;
	}

	randomize() {
		this.map(e => Math.floor(Math.random() * 10));
		return this;
	}

	print() {
		console.table(this.data);
		return this;
	}
}