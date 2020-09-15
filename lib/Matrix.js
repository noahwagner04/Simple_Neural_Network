class Matrix {
	constructor(rows, cols) {
		this.rows = rows;
		this.cols = cols;
		this.data = new Array(this.rows).fill(new Array(this.cols).fill(0));
	}
}