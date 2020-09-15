class Matrix {
	constructor(rows, cols) {
		this.rows = rows;
		this.cols = cols;
		this.data = new Array(this.rows).map(() => new Array(this.cols));
	}
}