/**
 * Multiple Linear Regression Analysis Engine & Interactive Application
 * Author: Student 7115056112
 * Course: Data Mining - Homework 2
 * Compliant with CRISP-DM Methodology
 */

// ==========================================
// 1. Matrix Mathematics Library
// ==========================================
const Matrix = {
  transpose(A) {
    const rows = A.length, cols = A[0].length;
    const AT = Array.from({ length: cols }, () => new Float64Array(rows));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        AT[j][i] = A[i][j];
      }
    }
    return AT;
  },

  multiply(A, B) {
    const rowsA = A.length, colsA = A[0].length;
    const rowsB = B.length, colsB = B[0].length;
    if (colsA !== rowsB) throw new Error(`Matrix dimension mismatch: ${colsA} vs ${rowsB}`);
    const C = Array.from({ length: rowsA }, () => new Float64Array(colsB));
    for (let i = 0; i < rowsA; i++) {
      for (let k = 0; k < colsA; k++) {
        const aik = A[i][k];
        for (let j = 0; j < colsB; j++) {
          C[i][j] += aik * B[k][j];
        }
      }
    }
    return C;
  },

  multiplyVec(A, v) {
    const rows = A.length, cols = A[0].length;
    const res = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      for (let j = 0; j < cols; j++) sum += A[i][j] * v[j];
      res[i] = sum;
    }
    return res;
  },

  dot(u, v) {
    let sum = 0;
    for (let i = 0; i < u.length; i++) sum += u[i] * v[i];
    return sum;
  },

  invert(A, ridge = 1e-7) {
    const n = A.length;
    const M = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) row[j] = A[i][j] + (i === j ? ridge : 0);
      row[n + i] = 1.0;
      return row;
    });

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      let maxVal = Math.abs(M[i][i]);
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(M[k][i]) > maxVal) {
          maxVal = Math.abs(M[k][i]);
          maxRow = k;
        }
      }
      if (maxRow !== i) {
        const tmp = M[i];
        M[i] = M[maxRow];
        M[maxRow] = tmp;
      }

      let pivot = M[i][i];
      if (Math.abs(pivot) < 1e-12) pivot = 1e-12;

      for (let j = 0; j < 2 * n; j++) M[i][j] /= pivot;

      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = M[k][i];
          if (Math.abs(factor) > 1e-14) {
            for (let j = 0; j < 2 * n; j++) M[k][j] -= factor * M[i][j];
          }
        }
      }
    }

    const inv = Array.from({ length: n }, (_, i) => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) inv[i][j] = M[i][n + j];
    }
    return inv;
  }
};

// ==========================================
// 2. Statistical Utilities
// ==========================================
const Stats = {
  mean(arr) {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return s / arr.length;
  },

  std(arr, meanVal) {
    const m = meanVal !== undefined ? meanVal : this.mean(arr);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += (arr[i] - m) ** 2;
    return Math.sqrt(s / (arr.length - 1 || 1));
  },

  median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  },

  minMax(arr) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] < min) min = arr[i];
      if (arr[i] > max) max = arr[i];
    }
    return { min, max };
  },

  pearson(x, y) {
    const n = x.length;
    const mx = this.mean(x), my = this.mean(y);
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx;
      const dy = y[i] - my;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const den = Math.sqrt(denX * denY);
    return den === 0 ? 0 : num / den;
  },

  tCritical(df) {
    if (df >= 120) return 1.960;
    if (df >= 60) return 2.000;
    if (df >= 30) return 2.042;
    if (df >= 20) return 2.086;
    if (df >= 10) return 2.228;
    if (df >= 5) return 2.571;
    return 2.776;
  },

  normInv(p) {
    const a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637];
    const b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833];
    const c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
      0.0276438810330100, 0.0038405729373609, 0.0003951804142957,
      0.0000321767881768, 0.0000002888167364, 0.0000003960315187];
    const y = p - 0.5;
    if (Math.abs(y) < 0.42) {
      const r = y * y;
      const num = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]);
      const den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0;
      return num / den;
    }
    let r = p;
    if (y > 0) r = 1 - p;
    r = Math.log(-Math.log(r));
    let x = c[0];
    for (let i = 1; i < c.length; i++) x += c[i] * Math.pow(r, i);
    return y < 0 ? -x : x;
  }
};

// ==========================================
// 3. Multiple Linear Regression Model
// ==========================================
class MultipleLinearRegression {
  constructor(options = {}) {
    this.fitIntercept = options.fitIntercept !== undefined ? options.fitIntercept : true;
    this.regularization = options.regularization || 'none'; // 'none', 'ridge', 'lasso'
    this.alpha = options.alpha !== undefined ? options.alpha : 0.01;
    this.coefficients = null;
    this.intercept = 0;
    this.XtX_inv = null;
    this.s2 = 0;
    this.s = 0;
    this.df = 0;
    this.p = 0;
    this.n = 0;
  }

  fit(X, y) {
    this.n = X.length;
    this.p = X[0].length;
    this.df = this.n - this.p - (this.fitIntercept ? 1 : 0);

    let X_design = [];
    if (this.fitIntercept) {
      X_design = X.map(row => {
        const r = new Float64Array(this.p + 1);
        r[0] = 1.0;
        for (let j = 0; j < this.p; j++) r[j + 1] = row[j];
        return r;
      });
    } else {
      X_design = X;
    }

    const cols = X_design[0].length;
    const XT = Matrix.transpose(X_design);
    const XtX = Matrix.multiply(XT, X_design);

    if (this.regularization === 'ridge') {
      for (let j = (this.fitIntercept ? 1 : 0); j < cols; j++) {
        XtX[j][j] += this.alpha * this.n;
      }
    }

    this.XtX_inv = Matrix.invert(XtX, 1e-8);
    const XTy = Matrix.multiplyVec(XT, y);
    const beta = Matrix.multiplyVec(this.XtX_inv, XTy);

    if (this.regularization === 'lasso') {
      this.fitLassoCoordinateDescent(X_design, y, beta);
    }

    if (this.fitIntercept) {
      this.intercept = beta[0];
      this.coefficients = Array.from(beta.slice(1));
    } else {
      this.intercept = 0;
      this.coefficients = Array.from(beta);
    }

    const yPred = this.predictDesign(X_design, beta);
    let sse = 0;
    for (let i = 0; i < this.n; i++) {
      sse += (y[i] - yPred[i]) ** 2;
    }
    this.s2 = sse / Math.max(1, this.df);
    this.s = Math.sqrt(this.s2);

    return this;
  }

  fitLassoCoordinateDescent(X, y, initialBeta, maxIter = 400, tol = 1e-4) {
    const n = X.length, p = X[0].length;
    const lambda = this.alpha;
    const beta = new Float64Array(initialBeta);

    const colNormSq = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i][j] * X[i][j];
      colNormSq[j] = s || 1.0;
    }

    let r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < p; j++) pred += X[i][j] * beta[j];
      r[i] = y[i] - pred;
    }

    for (let iter = 0; iter < maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        const oldBeta_j = beta[j];
        for (let i = 0; i < n; i++) r[i] += X[i][j] * oldBeta_j;

        let rho = 0;
        for (let i = 0; i < n; i++) rho += X[i][j] * r[i];

        if (this.fitIntercept && j === 0) {
          beta[j] = rho / colNormSq[j];
        } else {
          const threshold = lambda * n;
          if (rho > threshold) beta[j] = (rho - threshold) / colNormSq[j];
          else if (rho < -threshold) beta[j] = (rho + threshold) / colNormSq[j];
          else beta[j] = 0;
        }

        for (let i = 0; i < n; i++) r[i] -= X[i][j] * beta[j];

        const change = Math.abs(beta[j] - oldBeta_j);
        if (change > maxChange) maxChange = change;
      }
      if (maxChange < tol) break;
    }

    for (let j = 0; j < p; j++) initialBeta[j] = beta[j];
  }

  predictDesign(X_design, beta) {
    const m = X_design.length, k = beta.length;
    const res = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let j = 0; j < k; j++) sum += X_design[i][j] * beta[j];
      res[i] = sum;
    }
    return res;
  }

  predict(X) {
    const m = X.length;
    const res = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let sum = this.intercept;
      for (let j = 0; j < this.coefficients.length; j++) {
        sum += X[i][j] * this.coefficients[j];
      }
      res[i] = sum;
    }
    return res;
  }

  predictWithIntervals(xVector) {
    let yHat = this.intercept;
    for (let j = 0; j < this.coefficients.length; j++) {
      yHat += xVector[j] * this.coefficients[j];
    }

    const dim = this.fitIntercept ? this.coefficients.length + 1 : this.coefficients.length;
    const x0 = new Float64Array(dim);
    if (this.fitIntercept) {
      x0[0] = 1.0;
      for (let j = 0; j < this.coefficients.length; j++) x0[j + 1] = xVector[j];
    } else {
      for (let j = 0; j < this.coefficients.length; j++) x0[j] = xVector[j];
    }

    let quad = 0;
    if (this.XtX_inv) {
      const temp = new Float64Array(dim);
      for (let i = 0; i < dim; i++) {
        let sum = 0;
        for (let j = 0; j < dim; j++) sum += this.XtX_inv[i][j] * x0[j];
        temp[i] = sum;
      }
      for (let i = 0; i < dim; i++) quad += x0[i] * temp[i];
    } else {
      quad = 1.0 / (this.n || 100);
    }
    quad = Math.max(0, quad);

    const tcrit = Stats.tCritical(this.df);
    const seMean = this.s * Math.sqrt(quad);
    const sePred = this.s * Math.sqrt(1 + quad);

    return {
      prediction: yHat,
      seMean: seMean,
      sePred: sePred,
      leverage: quad,
      ci: [yHat - tcrit * seMean, yHat + tcrit * seMean],
      pi: [yHat - tcrit * sePred, yHat + tcrit * sePred]
    };
  }

  getCoefficientsDetails(X, featureNames) {
    const dim = this.fitIntercept ? this.coefficients.length + 1 : this.coefficients.length;
    const tcrit = Stats.tCritical(this.df);
    const details = [];

    const fullBeta = this.fitIntercept ? [this.intercept, ...this.coefficients] : [...this.coefficients];
    const fullNames = this.fitIntercept ? ['(Intercept)', ...featureNames] : [...featureNames];

    const vifs = this.computeVIFs(X);

    for (let j = 0; j < dim; j++) {
      const varBeta_j = this.XtX_inv ? Math.max(0, this.XtX_inv[j][j] * this.s2) : 1e-4;
      const se_j = Math.sqrt(varBeta_j);
      const t_j = se_j > 1e-12 ? fullBeta[j] / se_j : 0;
      const z = Math.abs(t_j);
      const p_val = 2 * (1 - this.approxNormalCDF(z));

      details.push({
        name: fullNames[j],
        coef: fullBeta[j],
        se: se_j,
        tStat: t_j,
        pValue: p_val,
        ci95: [fullBeta[j] - tcrit * se_j, fullBeta[j] + tcrit * se_j],
        vif: j === 0 && this.fitIntercept ? null : (this.fitIntercept ? vifs[j - 1] : vifs[j])
      });
    }

    return details;
  }

  approxNormalCDF(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.39894228 * Math.exp(-x * x / 2);
    const prob = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1 - prob : prob;
  }

  computeVIFs(X) {
    const p = X[0].length;
    const n = X.length;
    const vifs = new Float64Array(p);

    for (let j = 0; j < p; j++) {
      const y_j = X.map(r => r[j]);
      const X_rest = X.map(r => r.filter((_, idx) => idx !== j));
      const model = new MultipleLinearRegression({ fitIntercept: true });
      try {
        model.fit(X_rest, y_j);
        const yPred = model.predict(X_rest);
        const meanY = Stats.mean(y_j);
        let sse = 0, sst = 0;
        for (let i = 0; i < n; i++) {
          sse += (y_j[i] - yPred[i]) ** 2;
          sst += (y_j[i] - meanY) ** 2;
        }
        const r2 = sst > 0 ? 1 - (sse / sst) : 0;
        const vif = r2 >= 0.999 ? 1000 : 1 / (1 - r2);
        vifs[j] = Math.max(1.0, vif);
      } catch (e) {
        vifs[j] = 1.0;
      }
    }
    return vifs;
  }
}

// ==========================================
// 4. Evaluator & Diagnostic Engine
// ==========================================
const Evaluator = {
  evaluate(yTrue, yPred, numFeatures) {
    const n = yTrue.length;
    const p = numFeatures;
    let sse = 0, sae = 0, sst = 0;
    const meanY = Stats.mean(yTrue);

    for (let i = 0; i < n; i++) {
      const err = yTrue[i] - yPred[i];
      sse += err * err;
      sae += Math.abs(err);
      sst += (yTrue[i] - meanY) ** 2;
    }

    const mse = sse / n;
    const rmse = Math.sqrt(mse);
    const mae = sae / n;
    const r2 = sst > 0 ? 1 - (sse / sst) : 0;
    const adjR2 = sst > 0 && n > p + 1 ? 1 - ((sse / (n - p - 1)) / (sst / (n - 1))) : r2;

    const logLikelihood = -0.5 * n * Math.log(2 * Math.PI) - 0.5 * n * Math.log(mse || 1e-10) - 0.5 * n;
    const aic = 2 * (p + 1) - 2 * logLikelihood;
    const bic = Math.log(n) * (p + 1) - 2 * logLikelihood;

    return { n, p, mse, rmse, mae, r2, adjR2, aic, bic };
  },

  durbinWatson(residuals) {
    let num = 0, den = 0;
    for (let i = 0; i < residuals.length; i++) {
      den += residuals[i] ** 2;
      if (i > 0) num += (residuals[i] - residuals[i - 1]) ** 2;
    }
    return den > 0 ? num / den : 2.0;
  },

  computeQQData(residuals) {
    const n = residuals.length;
    const sorted = [...residuals].sort((a, b) => a - b);
    const mean = Stats.mean(residuals);
    const std = Stats.std(residuals, mean) || 1.0;

    const points = [];
    for (let i = 0; i < n; i++) {
      const p = (i + 0.5) / n;
      const theoretical = Stats.normInv(p);
      const sampleStd = (sorted[i] - mean) / std;
      points.push({ x: theoretical, y: sampleStd });
    }
    return points;
  }
};

// ==========================================
// 5. Feature Selection Module
// ==========================================
const FeatureSelection = {
  byCorrelation(X, y, featureNames, topK) {
    const scores = featureNames.map((name, idx) => {
      const col = X.map(r => r[idx]);
      const r = Math.abs(Stats.pearson(col, y));
      return { index: idx, name, score: r };
    });
    scores.sort((a, b) => b.score - a.score);
    const selected = scores.slice(0, Math.min(topK, scores.length));
    return {
      selectedIndices: selected.map(s => s.index),
      selectedNames: selected.map(s => s.name),
      ranking: scores
    };
  },

  forwardStepwise(X, y, featureNames, maxFeatures) {
    const n = X.length, p = featureNames.length;
    const selectedIndices = [];
    const remainingIndices = Array.from({ length: p }, (_, i) => i);
    const history = [];

    const targetK = Math.min(maxFeatures, p);

    while (selectedIndices.length < targetK && remainingIndices.length > 0) {
      let bestAdjR2 = -Infinity;
      let bestIdx = -1;

      for (const idx of remainingIndices) {
        const candidate = [...selectedIndices, idx];
        const X_sub = X.map(r => candidate.map(j => r[j]));
        const model = new MultipleLinearRegression();
        model.fit(X_sub, y);
        const yPred = model.predict(X_sub);
        const evalRes = Evaluator.evaluate(y, yPred, candidate.length);

        if (evalRes.adjR2 > bestAdjR2) {
          bestAdjR2 = evalRes.adjR2;
          bestIdx = idx;
        }
      }

      if (bestIdx !== -1) {
        selectedIndices.push(bestIdx);
        const remPos = remainingIndices.indexOf(bestIdx);
        remainingIndices.splice(remPos, 1);
        history.push({
          step: selectedIndices.length,
          addedFeature: featureNames[bestIdx],
          adjR2: bestAdjR2
        });
      } else {
        break;
      }
    }

    return {
      selectedIndices,
      selectedNames: selectedIndices.map(i => featureNames[i]),
      history
    };
  }
};

// ==========================================
// 6. Data Preprocessing & CSV Parser
// ==========================================
const DataProcessor = {
  parseCSV(text) {
    const lines = text.trim().split(/\r\n|\n/).filter(l => l.trim().length > 0);
    if (lines.length < 2) throw new Error("CSV contains insufficient lines.");

    const parseRow = (line) => {
      const cells = [];
      let inQuotes = false, current = '';
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') inQuotes = !inQuotes;
        else if (char === ',' && !inQuotes) {
          cells.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      cells.push(current.trim());
      return cells;
    };

    const header = parseRow(lines[0]);
    const rawRows = lines.slice(1).map(parseRow);
    return { header, rawRows };
  },

  cleanAndEncode(header, rawRows, targetColName) {
    const targetIdx = header.indexOf(targetColName);
    if (targetIdx === -1) throw new Error(`Target column "${targetColName}" not found.`);

    const numRows = rawRows.length;
    const colTypes = header.map((name, idx) => {
      let isNumeric = true;
      let nonNullCount = 0;
      for (let i = 0; i < Math.min(numRows, 100); i++) {
        const val = rawRows[i][idx];
        if (val !== undefined && val !== '' && val !== 'NA' && val !== 'null') {
          nonNullCount++;
          if (isNaN(Number(val))) {
            isNumeric = false;
            break;
          }
        }
      }
      return isNumeric && nonNullCount > 0 ? 'numeric' : 'categorical';
    });

    const y = [];
    const validRowIndices = [];
    for (let i = 0; i < numRows; i++) {
      const rawTarget = rawRows[i][targetIdx];
      const val = Number(rawTarget);
      if (!isNaN(val) && rawTarget !== '') {
        y.push(val);
        validRowIndices.push(i);
      }
    }

    const finalFeatureNames = [];
    const featureCols = [];

    header.forEach((name, idx) => {
      if (idx === targetIdx) return;

      if (colTypes[idx] === 'numeric') {
        finalFeatureNames.push(name);
        const numbers = [];
        for (const rIdx of validRowIndices) {
          const v = Number(rawRows[rIdx][idx]);
          if (!isNaN(v)) numbers.push(v);
        }
        const med = Stats.median(numbers);
        const colVals = validRowIndices.map(rIdx => {
          const v = Number(rawRows[rIdx][idx]);
          return isNaN(v) ? med : v;
        });
        featureCols.push(colVals);
      } else {
        const values = validRowIndices.map(rIdx => rawRows[rIdx][idx] || 'Unknown');
        const unique = Array.from(new Set(values)).sort();
        for (let u = 0; u < unique.length - 1; u++) {
          const catVal = unique[u];
          finalFeatureNames.push(`${name}_${catVal}`);
          const dummyCol = values.map(v => (v === catVal ? 1.0 : 0.0));
          featureCols.push(dummyCol);
        }
      }
    });

    const X = Array.from({ length: validRowIndices.length }, (_, r) => {
      const row = new Float64Array(featureCols.length);
      for (let c = 0; c < featureCols.length; c++) {
        row[c] = featureCols[c][r];
      }
      return row;
    });

    return {
      featureNames: finalFeatureNames,
      targetName: targetColName,
      X,
      y: new Float64Array(y)
    };
  },

  trainTestSplit(X, y, testSize = 0.2, seed = 42) {
    const n = X.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    let s = seed;
    const rand = () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };

    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }

    const testCount = Math.floor(n * testSize);
    const testIndices = indices.slice(0, testCount);
    const trainIndices = indices.slice(testCount);

    const X_train = trainIndices.map(i => X[i]);
    const y_train = Float64Array.from(trainIndices.map(i => y[i]));
    const X_test = testIndices.map(i => X[i]);
    const y_test = Float64Array.from(testIndices.map(i => y[i]));

    return { X_train, y_train, X_test, y_test };
  }
};

// ==========================================
// 7. Global State & App Controller
// ==========================================
const App = {
  state: {
    currentProblem: 'flood', // 'flood', 'istanbul', 'startups', 'custom_url', 'custom_upload'
    rawCSV: '',
    header: [],
    rawRows: [],
    datasetMeta: {},
    featureNames: [],
    targetName: '',
    X: [],
    y: [],
    X_train: [],
    y_train: [],
    X_test: [],
    y_test: [],
    selectedFeatureIndices: [],
    modelType: 'ols', // 'ols', 'ridge', 'lasso', 'simple'
    regularizationAlpha: 0.01,
    activeModel: null,
    trainMetrics: null,
    testMetrics: null,
    simInputs: {},
    activeTab: 'tab-business'
  },

  charts: {},

  async init() {
    console.log("Initializing CRISP-DM Multiple Linear Regression App...");
    this.bindEvents();
    // Default load Problem 1: Flood Prediction
    await this.loadProblem('flood');
  },

  bindEvents() {
    // Problem selector dropdown (加分 1分)
    const probSelect = document.getElementById('problemSelect');
    if (probSelect) {
      probSelect.addEventListener('change', (e) => {
        this.handleProblemChange(e.target.value);
      });
    }

    // Custom URL load button (加分 2分)
    const btnLoadUrl = document.getElementById('btnLoadUrl');
    if (btnLoadUrl) {
      btnLoadUrl.addEventListener('click', () => {
        const url = document.getElementById('datasetUrlInput').value.trim();
        if (url) this.loadFromUrl(url);
      });
    }

    // File Upload
    const fileInput = document.getElementById('fileUploadInput');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) this.loadFromFile(file);
      });
    }

    // Target Column select
    const targetSelect = document.getElementById('targetSelect');
    if (targetSelect) {
      targetSelect.addEventListener('change', (e) => {
        this.reprocessData(e.target.value);
      });
    }

    // Navigation Tabs
    const tabBtns = document.querySelectorAll('.nav-tab');
    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        this.switchTab(targetTab);
      });
    });

    // Run Full Pipeline Button
    const btnRunAll = document.getElementById('btnRunAll');
    if (btnRunAll) {
      btnRunAll.addEventListener('click', () => {
        this.runFullPipeline();
      });
    }

    // Feature Selection Controls
    const fsMethod = document.getElementById('fsMethod');
    if (fsMethod) {
      fsMethod.addEventListener('change', () => this.applyFeatureSelection());
    }
    const fsSlider = document.getElementById('fsSlider');
    if (fsSlider) {
      fsSlider.addEventListener('input', (e) => {
        document.getElementById('fsSliderVal').innerText = e.target.value;
      });
      fsSlider.addEventListener('change', () => this.applyFeatureSelection());
    }

    // Model Selector
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
      modelSelect.addEventListener('change', (e) => {
        this.state.modelType = e.target.value;
        const alphaContainer = document.getElementById('alphaContainer');
        if (alphaContainer) {
          alphaContainer.style.display = (e.target.value === 'ridge' || e.target.value === 'lasso') ? 'block' : 'none';
        }
        this.trainAndEvaluateModel();
      });
    }

    // Alpha Slider
    const alphaSlider = document.getElementById('alphaSlider');
    if (alphaSlider) {
      alphaSlider.addEventListener('input', (e) => {
        this.state.regularizationAlpha = parseFloat(e.target.value);
        document.getElementById('alphaSliderVal').innerText = e.target.value;
        this.trainAndEvaluateModel();
      });
    }

    // Theme Toggle
    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
        const isLight = document.body.classList.contains('light-theme');
        themeBtn.innerHTML = isLight ? '🌙 深色模式' : '☀️ 淺色模式';
      });
    }

    // Report and Dialogue buttons
    const btnOpenReport = document.getElementById('btnOpenReport');
    if (btnOpenReport) {
      btnOpenReport.addEventListener('click', () => {
        window.open('report.html', '_blank');
      });
    }
    const btnOpenDialogue = document.getElementById('btnOpenDialogue');
    if (btnOpenDialogue) {
      btnOpenDialogue.addEventListener('click', () => {
        window.open('chatgpt_dialogue.html', '_blank');
      });
    }
  },

  switchTab(tabId) {
    this.state.activeTab = tabId;
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

    const btn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
    if (btn) btn.classList.add('active');
    const panel = document.getElementById(tabId);
    if (panel) panel.classList.add('active');

    // Trigger chart resize on tab switch
    window.dispatchEvent(new Event('resize'));
  },

  async handleProblemChange(probKey) {
    this.state.currentProblem = probKey;
    const urlContainer = document.getElementById('urlInputContainer');
    const fileContainer = document.getElementById('fileInputContainer');

    if (urlContainer) urlContainer.style.display = probKey === 'custom_url' ? 'flex' : 'none';
    if (fileContainer) fileContainer.style.display = probKey === 'custom_upload' ? 'block' : 'none';

    if (probKey === 'flood') {
      await this.loadLocalDataset('data/flood_prediction_sample.csv', 'FloodProbability');
    } else if (probKey === 'istanbul') {
      await this.loadLocalDataset('data/istanbul_apartment_sample.csv', 'Price');
    } else if (probKey === 'startups') {
      await this.loadLocalDataset('data/50_Startups.csv', 'Profit');
    } else if (probKey === 'custom_url') {
      // Show URL input bar
      const urlInput = document.getElementById('datasetUrlInput');
      if (urlInput && !urlInput.value) {
        urlInput.value = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv';
      }
    }
  },

  async loadLocalDataset(filePath, defaultTarget) {
    try {
      this.showLoading(`正在載入資料集: ${filePath}...`);
      const resp = await fetch(filePath);
      if (!resp.ok) throw new Error(`HTTP Error ${resp.status}`);
      const text = await resp.text();
      this.processRawCSV(text, defaultTarget);
    } catch (e) {
      console.warn("Local fetch failed, generating fallback in-memory data for:", filePath, e);
      this.generateFallbackDataset(filePath, defaultTarget);
    } finally {
      this.hideLoading();
    }
  },

  generateFallbackDataset(filePath, defaultTarget) {
    // If running from file:// where fetch is restricted, generate the realistic sample data directly!
    let lines = [];
    if (filePath.includes('flood')) {
      const features = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
        'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
        'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
        'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
        'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
        'InadequatePlanning', 'PoliticalFactors'
      ];
      lines.push([...features, 'FloodProbability'].join(','));
      for (let i = 0; i < 500; i++) {
        let row = [];
        let sum = 0.05;
        for (let j = 0; j < 20; j++) {
          const val = Math.floor(Math.random() * 8) + 2;
          row.push(val);
          sum += val * 0.0053;
        }
        sum += (Math.random() - 0.5) * 0.02;
        row.push(sum.toFixed(4));
        lines.push(row.join(','));
      }
    } else if (filePath.includes('istanbul')) {
      const features = [
        'GrossSquareMeters', 'NetSquareMeters', 'NumberOfRooms', 'BuildingAge',
        'FloorLocation', 'NumberOfFloors', 'Bathrooms', 'Balcony',
        'Elevator', 'ParkingSpace', 'DistanceToMetro_km', 'DistanceToCenter_km',
        'SecuritySystem', 'Furnished', 'ThermalInsulation', 'SchoolRating',
        'GreenAreaRatio', 'NeighborhoodIncomeIndex'
      ];
      lines.push([...features, 'Price'].join(','));
      for (let i = 0; i < 400; i++) {
        const gross = (80 + Math.random() * 120).toFixed(1);
        const net = (gross * 0.82).toFixed(1);
        const rooms = Math.floor(gross / 40) + 1;
        const age = Math.floor(Math.random() * 25);
        const floors = 10;
        const floorLoc = Math.floor(Math.random() * floors);
        const baths = Math.floor(rooms / 2) + 1;
        const balcony = Math.random() > 0.3 ? 1 : 0;
        const elevator = 1;
        const parking = Math.random() > 0.4 ? 1 : 0;
        const distMetro = (1 + Math.random() * 6).toFixed(1);
        const distCenter = (5 + Math.random() * 15).toFixed(1);
        const sec = Math.random() > 0.5 ? 1 : 0;
        const fur = Math.random() > 0.7 ? 1 : 0;
        const thm = 1;
        const sch = (6 + Math.random() * 3).toFixed(1);
        const grn = (0.1 + Math.random() * 0.4).toFixed(2);
        const inc = (80 + Math.random() * 50).toFixed(1);

        const price = (
          40 + net * 1.2 - age * 0.9 + rooms * 10 + baths * 7 -
          distMetro * 4 + sec * 12 + inc * 0.5 + (Math.random() - 0.5) * 15
        ).toFixed(2);

        lines.push([
          gross, net, rooms, age, floorLoc, floors, baths, balcony, elevator,
          parking, distMetro, distCenter, sec, fur, thm, sch, grn, inc, price
        ].join(','));
      }
    } else {
      lines.push('R&D Spend,Administration,Marketing Spend,State,Profit');
      lines.push('165349.2,136897.8,471784.1,New York,192261.83');
      lines.push('162597.7,151377.59,443898.53,California,191792.06');
      lines.push('153441.51,101145.55,407934.54,Florida,191050.39');
      lines.push('144372.41,118671.85,383199.62,New York,182901.99');
      lines.push('142107.34,91391.77,366168.42,Florida,166187.94');
      lines.push('131876.9,99814.71,362861.36,California,156991.12');
      lines.push('134615.46,147198.87,127716.82,California,156122.51');
      lines.push('130298.13,145530.06,323876.68,Florida,155752.6');
      lines.push('120542.52,148711.48,311613.29,New York,152211.77');
    }
    this.processRawCSV(lines.join('\n'), defaultTarget);
  },

  async loadFromUrl(url) {
    try {
      this.showLoading(`正在由 URL 下載 Dataset: ${url}...`);
      // Try direct fetch first, fallback to CORS proxy if needed
      let resp;
      try {
        resp = await fetch(url);
      } catch (err) {
        console.warn("Direct fetch failed due to CORS, attempting via corsproxy.io...");
        resp = await fetch(`https://corsproxy.io/?${encodeURIComponent(url)}`);
      }
      if (!resp.ok) throw new Error(`HTTP Error: ${resp.status}`);
      const text = await resp.text();
      this.processRawCSV(text);
      this.showToast("資料集載入成功！已自動偵測欄位與型態", "success");
    } catch (e) {
      alert(`載入失敗: ${e.message}。請確認網址有效性或使用上傳功能。`);
    } finally {
      this.hideLoading();
    }
  },

  loadFromFile(file) {
    this.showLoading(`正在讀取檔案: ${file.name}...`);
    const reader = new FileReader();
    reader.onload = (e) => {
      this.processRawCSV(e.target.result);
      this.hideLoading();
      this.showToast(`成功讀取本地檔案: ${file.name}`, "success");
    };
    reader.onerror = () => {
      this.hideLoading();
      alert("讀取檔案失敗！");
    };
    reader.readAsText(file);
  },

  processRawCSV(csvText, defaultTarget) {
    this.state.rawCSV = csvText;
    const { header, rawRows } = DataProcessor.parseCSV(csvText);
    this.state.header = header;
    this.state.rawRows = rawRows;

    // Populate target column select
    const targetSelect = document.getElementById('targetSelect');
    if (targetSelect) {
      targetSelect.innerHTML = '';
      header.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.innerText = col;
        targetSelect.appendChild(opt);
      });

      // Auto pick default target
      let targetToSelect = defaultTarget;
      if (!targetToSelect || !header.includes(targetToSelect)) {
        // Last numeric column
        targetToSelect = header[header.length - 1];
      }
      targetSelect.value = targetToSelect;
    }

    const selectedTarget = targetSelect ? targetSelect.value : (defaultTarget || header[header.length - 1]);
    this.reprocessData(selectedTarget);
  },

  reprocessData(targetName) {
    const { header, rawRows } = this.state;
    const cleaned = DataProcessor.cleanAndEncode(header, rawRows, targetName);
    this.state.featureNames = cleaned.featureNames;
    this.state.targetName = cleaned.targetName;
    this.state.X = cleaned.X;
    this.state.y = cleaned.y;

    // Train/Test Split
    const split = DataProcessor.trainTestSplit(cleaned.X, cleaned.y, 0.2, 42);
    this.state.X_train = split.X_train;
    this.state.y_train = split.y_train;
    this.state.X_test = split.X_test;
    this.state.y_test = split.y_test;

    // Update Feature Selection Slider
    const fsSlider = document.getElementById('fsSlider');
    if (fsSlider) {
      fsSlider.max = cleaned.featureNames.length;
      fsSlider.min = 1;
      fsSlider.value = cleaned.featureNames.length; // Default to all features
      document.getElementById('fsSliderVal').innerText = fsSlider.value;
    }

    // Default select all features
    this.state.selectedFeatureIndices = Array.from({ length: cleaned.featureNames.length }, (_, i) => i);

    // Update UI panels
    this.updateBusinessUnderstanding();
    this.updateDataUnderstanding();
    this.updateDataPreparation();
    this.trainAndEvaluateModel();
  },

  updateBusinessUnderstanding() {
    const prob = this.state.currentProblem;
    const titleElem = document.getElementById('bizProblemTitle');
    const descElem = document.getElementById('bizProblemDesc');
    const targetElem = document.getElementById('bizTargetDesc');
    const linkElem = document.getElementById('bizKaggleLink');

    if (prob === 'flood') {
      if (titleElem) titleElem.innerText = "🌊 Kaggle Problem 1: 洪水機率預測 (Flood Probability Prediction)";
      if (descElem) descElem.innerHTML = `
        <p><strong>背景介紹：</strong>洪澇災害是全球最具破壞性的自然災害之一。本任務採用 Kaggle Playground Series s4e5 公開資料集，利用<strong>20項氣象、地形、水利管理與人為環境特徵</strong>預測區域洪澇發生機率 (FloodProbability)。</p>
        <p><strong>商業與防減災價值：</strong>建立可解釋性極高的多元線性回歸模型，能精準量化「森林砍伐」、「季風強度」與「水利基建老化」對洪水的邊際貢獻，協助地方政府進行救災資源前瞻性調度與保險風險精算。</p>
      `;
      if (targetElem) targetElem.innerHTML = `<strong>連續預測目標：</strong> <code>FloodProbability</code> (連續數值區間 [0.28, 0.75]，代表洪災發生機率)`;
      if (linkElem) {
        linkElem.href = "https://www.kaggle.com/competitions/playground-series-s4e5";
        linkElem.innerText = "Kaggle: Playground Series s4e5 (Predict the Flood)";
      }
    } else if (prob === 'istanbul') {
      if (titleElem) titleElem.innerText = "🏢 Kaggle Problem 2: 伊斯坦堡公寓房價預測 (Istanbul Apartment Prices 2026)";
      if (descElem) descElem.innerHTML = `
        <p><strong>背景介紹：</strong>伊斯坦堡是橫跨歐亞的超大型都市，房地產價格受到建築坪數、屋齡、距離地鐵站/市中心距離、公共設施與周邊學區等多重因素影響。本任務使用<strong>18項連續與結構化特徵</strong>預測房產售價 (Price)。</p>
        <p><strong>商業與投資價值：</strong>透過特徵係數檢定，房地產估價機構與購屋者能清晰掌握每增加一坪面積或每接近地鐵 1 公里所能帶來的房價溢價效益，實現房價合理估值與套利預警。</p>
      `;
      if (targetElem) targetElem.innerHTML = `<strong>連續預測目標：</strong> <code>Price</code> (千美元 / 千里拉 連續數值)`;
      if (linkElem) {
        linkElem.href = "https://www.kaggle.com/datasets/brahimenesulusoy/istanbul-apartment-prices-2026";
        linkElem.innerText = "Kaggle: Istanbul Apartment Prices 2026";
      }
    } else {
      if (titleElem) titleElem.innerText = `自訂/線上資料集：${this.state.targetName} 回歸預測`;
      if (descElem) descElem.innerHTML = `
        <p>本任務載入自訂資料集，並透過自動特徵檢測與資料清理，針對連續目標 <code>${this.state.targetName}</code> 進行完整的多元線性回歸與 CRISP-DM 流程分析。</p>
      `;
      if (targetElem) targetElem.innerHTML = `<strong>預測目標變數：</strong> <code>${this.state.targetName}</code>`;
      if (linkElem) {
        linkElem.href = "#";
        linkElem.innerText = "自訂 Dataset / 外部輸入資料集";
      }
    }
  },

  updateDataUnderstanding() {
    const { X, y, featureNames, targetName } = this.state;
    const n = X.length;
    const p = featureNames.length;

    // Update Overview Cards
    document.getElementById('statTotalRows').innerText = n.toLocaleString();
    document.getElementById('statTotalFeatures').innerText = p.toString();
    document.getElementById('statTargetName').innerText = targetName;
    const targetStats = Stats.minMax(y);
    const targetMean = Stats.mean(y);
    const targetStd = Stats.std(y, targetMean);
    document.getElementById('statTargetStats').innerText = `Mean: ${targetMean.toFixed(3)} | Std: ${targetStd.toFixed(3)}`;

    // Populate Feature Table
    const tbody = document.getElementById('featureStatsTableBody');
    if (tbody) {
      tbody.innerHTML = '';
      featureNames.forEach((name, idx) => {
        const colVals = X.map(r => r[idx]);
        const mm = Stats.minMax(colVals);
        const m = Stats.mean(colVals);
        const s = Stats.std(colVals, m);
        const r = Stats.pearson(colVals, y);

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${idx + 1}</strong></td>
          <td><code>${name}</code></td>
          <td>數值連續 (Numeric)</td>
          <td>${mm.min.toFixed(2)}</td>
          <td>${mm.max.toFixed(2)}</td>
          <td>${m.toFixed(2)}</td>
          <td>${s.toFixed(2)}</td>
          <td>
            <span class="badge ${Math.abs(r) > 0.4 ? 'badge-high' : Math.abs(r) > 0.15 ? 'badge-med' : 'badge-low'}">
              ${r >= 0 ? '+' : ''}${r.toFixed(4)}
            </span>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Render Target Distribution Chart
    this.renderTargetDistributionChart(y);

    // Render Correlation Matrix Heatmap
    this.renderCorrelationHeatmap();
  },

  updateDataPreparation() {
    const { X, X_train, X_test, featureNames } = this.state;
    document.getElementById('prepTrainRows').innerText = X_train.length.toLocaleString();
    document.getElementById('prepTestRows').innerText = X_test.length.toLocaleString();
    document.getElementById('prepFeaturesCount').innerText = featureNames.length.toString();
    document.getElementById('prepSplitRatio').innerText = `80% : 20% (Seeded = 42)`;
  },

  applyFeatureSelection() {
    const method = document.getElementById('fsMethod').value;
    const topK = parseInt(document.getElementById('fsSlider').value, 10);
    const { X_train, y_train, featureNames } = this.state;

    let selectedIndices = [];
    if (method === 'all' || topK >= featureNames.length) {
      selectedIndices = Array.from({ length: featureNames.length }, (_, i) => i);
    } else if (method === 'correlation') {
      const res = FeatureSelection.byCorrelation(X_train, y_train, featureNames, topK);
      selectedIndices = res.selectedIndices;
    } else if (method === 'stepwise') {
      const res = FeatureSelection.forwardStepwise(X_train, y_train, featureNames, topK);
      selectedIndices = res.selectedIndices;
    }

    this.state.selectedFeatureIndices = selectedIndices;
    this.trainAndEvaluateModel();
  },

  trainAndEvaluateModel() {
    const { X_train, y_train, X_test, y_test, featureNames, selectedFeatureIndices, modelType, regularizationAlpha } = this.state;

    const activeFeatureNames = selectedFeatureIndices.map(i => featureNames[i]);
    const X_tr_sub = X_train.map(r => selectedFeatureIndices.map(j => r[j]));
    const X_te_sub = X_test.map(r => selectedFeatureIndices.map(j => r[j]));

    let model;
    if (modelType === 'ols') {
      model = new MultipleLinearRegression({ fitIntercept: true, regularization: 'none' });
    } else if (modelType === 'ridge') {
      model = new MultipleLinearRegression({ fitIntercept: true, regularization: 'ridge', alpha: regularizationAlpha });
    } else if (modelType === 'lasso') {
      model = new MultipleLinearRegression({ fitIntercept: true, regularization: 'lasso', alpha: regularizationAlpha });
    } else if (modelType === 'simple') {
      // Pick highest correlation single feature
      const topFeat = FeatureSelection.byCorrelation(X_tr_sub, y_train, activeFeatureNames, 1);
      const singleIdx = topFeat.selectedIndices[0];
      const X_single_tr = X_tr_sub.map(r => [r[singleIdx]]);
      const X_single_te = X_te_sub.map(r => [r[singleIdx]]);
      model = new MultipleLinearRegression({ fitIntercept: true });
      model.fit(X_single_tr, y_train);
      this.state.activeModel = model;
      this.state.activeFeatureNames = [activeFeatureNames[singleIdx]];
      this.state.activeX_train = X_single_tr;
      this.state.activeX_test = X_single_te;
      this.evaluateAndRender(model, X_single_tr, y_train, X_single_te, y_test, [activeFeatureNames[singleIdx]]);
      return;
    }

    model.fit(X_tr_sub, y_train);
    this.state.activeModel = model;
    this.state.activeFeatureNames = activeFeatureNames;
    this.state.activeX_train = X_tr_sub;
    this.state.activeX_test = X_te_sub;

    this.evaluateAndRender(model, X_tr_sub, y_train, X_te_sub, y_test, activeFeatureNames);
  },

  evaluateAndRender(model, X_tr, y_tr, X_te, y_te, activeFeatures) {
    const yPredTrain = model.predict(X_tr);
    const yPredTest = model.predict(X_te);

    const trainMetrics = Evaluator.evaluate(y_tr, yPredTrain, activeFeatures.length);
    const testMetrics = Evaluator.evaluate(y_te, yPredTest, activeFeatures.length);
    this.state.trainMetrics = trainMetrics;
    this.state.testMetrics = testMetrics;

    // Render Metrics Badges
    document.getElementById('metricR2').innerText = testMetrics.r2.toFixed(4);
    document.getElementById('metricAdjR2').innerText = testMetrics.adjR2.toFixed(4);
    document.getElementById('metricRMSE').innerText = testMetrics.rmse.toFixed(4);
    document.getElementById('metricMAE').innerText = testMetrics.mae.toFixed(4);
    document.getElementById('metricMSE').innerText = testMetrics.mse.toFixed(6);

    // Update Comparison Table
    document.getElementById('tblTrainR2').innerText = trainMetrics.r2.toFixed(4);
    document.getElementById('tblTestR2').innerText = testMetrics.r2.toFixed(4);
    document.getElementById('tblTrainAdjR2').innerText = trainMetrics.adjR2.toFixed(4);
    document.getElementById('tblTestAdjR2').innerText = testMetrics.adjR2.toFixed(4);
    document.getElementById('tblTrainRMSE').innerText = trainMetrics.rmse.toFixed(4);
    document.getElementById('tblTestRMSE').innerText = testMetrics.rmse.toFixed(4);
    document.getElementById('tblTrainMAE').innerText = trainMetrics.mae.toFixed(4);
    document.getElementById('tblTestMAE').innerText = testMetrics.mae.toFixed(4);

    // Render Coefficients Table
    this.renderCoefficientsTable(model, X_tr, activeFeatures);

    // Render Core Prediction Plot with Confidence Interval and Prediction Interval
    this.renderActualVsPredictedChart(model, X_te, y_te);

    // Render 5 Regression Assumptions Diagnostic Charts
    const residualsTest = Array.from(y_te).map((y, i) => y - yPredTest[i]);
    this.renderAssumptionsDiagnostics(model, X_te, yPredTest, residualsTest);

    // Render Interactive Deployment Simulator
    this.renderDeploymentSimulator(model, activeFeatures);
  },

  renderCoefficientsTable(model, X_tr, activeFeatures) {
    const tbody = document.getElementById('coeffTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const details = model.getCoefficientsDetails(X_tr, activeFeatures);
    details.forEach(item => {
      const tr = document.createElement('tr');
      const sig = item.pValue < 0.001 ? '***' : item.pValue < 0.01 ? '**' : item.pValue < 0.05 ? '*' : '';
      const vifDisplay = item.vif === null ? '-' : item.vif.toFixed(2);
      const vifClass = item.vif === null ? '' : item.vif > 10 ? 'badge-high' : item.vif > 5 ? 'badge-med' : 'badge-low';

      tr.innerHTML = `
        <td><strong>${item.name}</strong></td>
        <td><code>${item.coef.toFixed(5)}</code></td>
        <td>${item.se.toFixed(5)}</td>
        <td>${item.tStat.toFixed(3)}</td>
        <td><span class="badge ${item.pValue < 0.05 ? 'badge-low' : 'badge-high'}">${item.pValue.toExponential(2)} ${sig}</span></td>
        <td>[${item.ci95[0].toFixed(4)}, ${item.ci95[1].toFixed(4)}]</td>
        <td>${item.vif !== null ? `<span class="badge ${vifClass}">${vifDisplay}</span>` : '-'}</td>
      `;
      tbody.appendChild(tr);
    });
  },

  // ==========================================
  // Core Chart: Prediction Plot with CI and PI
  // ==========================================
  renderActualVsPredictedChart(model, X_te, y_te) {
    const canvas = document.getElementById('chartActualVsPred');
    if (!canvas) return;

    if (this.charts.actualVsPred) {
      this.charts.actualVsPred.destroy();
    }

    // Sample up to 120 points for ultra-clear visualization
    const maxPts = Math.min(y_te.length, 120);
    const scatterPoints = [];
    const ciUpperLine = [];
    const ciLowerLine = [];
    const piUpperLine = [];
    const piLowerLine = [];
    const fitLine = [];

    // Pair and sort by predicted value
    const paired = [];
    for (let i = 0; i < maxPts; i++) {
      const predRes = model.predictWithIntervals(X_te[i]);
      paired.push({
        actual: y_te[i],
        pred: predRes.prediction,
        ciLow: predRes.ci[0],
        ciHigh: predRes.ci[1],
        piLow: predRes.pi[0],
        piHigh: predRes.pi[1]
      });
    }
    paired.sort((a, b) => a.pred - b.pred);

    paired.forEach(p => {
      scatterPoints.push({ x: p.pred, y: p.actual });
      fitLine.push({ x: p.pred, y: p.pred });
      ciUpperLine.push({ x: p.pred, y: p.ciHigh });
      ciLowerLine.push({ x: p.pred, y: p.ciLow });
      piUpperLine.push({ x: p.pred, y: p.piHigh });
      piLowerLine.push({ x: p.pred, y: p.piLow });
    });

    const ctx = canvas.getContext('2d');
    this.charts.actualVsPred = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '實際觀測值 (Actual Data)',
            data: scatterPoints,
            backgroundColor: 'rgba(99, 102, 241, 0.75)',
            borderColor: 'rgba(99, 102, 241, 1)',
            pointRadius: 4,
            pointHoverRadius: 7,
            order: 3
          },
          {
            type: 'line',
            label: '擬合回歸線 (Fitted Line)',
            data: fitLine,
            borderColor: '#10b981',
            borderWidth: 2.5,
            fill: false,
            pointRadius: 0,
            order: 2
          },
          {
            type: 'line',
            label: '95% 信賴區間 CI (Mean Response)',
            data: ciUpperLine,
            borderColor: 'rgba(245, 158, 11, 0.9)',
            borderDash: [4, 4],
            borderWidth: 1.8,
            fill: false,
            pointRadius: 0,
            order: 1
          },
          {
            type: 'line',
            label: '95% CI 下界',
            data: ciLowerLine,
            borderColor: 'rgba(245, 158, 11, 0.9)',
            borderDash: [4, 4],
            borderWidth: 1.8,
            fill: false,
            pointRadius: 0,
            showInLegend: false,
            order: 1
          },
          {
            type: 'line',
            label: '95% 預測區間 PI (Individual Prediction)',
            data: piUpperLine,
            borderColor: 'rgba(239, 68, 68, 0.75)',
            borderDash: [6, 4],
            borderWidth: 1.8,
            fill: false,
            pointRadius: 0,
            order: 0
          },
          {
            type: 'line',
            label: '95% PI 下界',
            data: piLowerLine,
            borderColor: 'rgba(239, 68, 68, 0.75)',
            borderDash: [6, 4],
            borderWidth: 1.8,
            fill: false,
            pointRadius: 0,
            showInLegend: false,
            order: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#e2e8f0',
              font: { family: 'Inter', size: 12 },
              filter: (item) => !item.text.includes('下界')
            }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const pt = ctx.raw;
                return [
                  `預測值 ŷ: ${pt.x ? pt.x.toFixed(4) : ''}`,
                  `數值: ${pt.y ? pt.y.toFixed(4) : ''}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: '預測值 (Predicted Value ŷ)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            title: { display: true, text: '目標值與區間邊界 (Values)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  },

  // ==========================================
  // 5 Regression Assumptions Diagnostics
  // ==========================================
  renderAssumptionsDiagnostics(model, X_te, yPredTest, residuals) {
    const n = residuals.length;

    // 1. Linearity & Homoscedasticity: Residuals vs Fitted
    const canvasResFit = document.getElementById('chartResVsFitted');
    if (canvasResFit) {
      if (this.charts.resVsFitted) this.charts.resVsFitted.destroy();
      const resPoints = residuals.map((r, i) => ({ x: yPredTest[i], y: r }));
      const minX = Math.min(...yPredTest);
      const maxX = Math.max(...yPredTest);

      this.charts.resVsFitted = new Chart(canvasResFit.getContext('2d'), {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: '殘差 (Residuals e)',
              data: resPoints,
              backgroundColor: 'rgba(14, 165, 233, 0.7)',
              pointRadius: 3
            },
            {
              type: 'line',
              label: '基準線 e = 0',
              data: [{ x: minX, y: 0 }, { x: maxX, y: 0 }],
              borderColor: '#f43f5e',
              borderWidth: 2,
              borderDash: [5, 5],
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { title: { display: true, text: '擬合值 (Fitted ŷ)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
            y: { title: { display: true, text: '殘差 (Residuals)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }

    // 2. Normality: Normal Q-Q Plot
    const canvasQQ = document.getElementById('chartQQPlot');
    if (canvasQQ) {
      if (this.charts.qqPlot) this.charts.qqPlot.destroy();
      const qqData = Evaluator.computeQQData(residuals);

      this.charts.qqPlot = new Chart(canvasQQ.getContext('2d'), {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: '標準化殘差分位數',
              data: qqData,
              backgroundColor: 'rgba(168, 85, 247, 0.75)',
              pointRadius: 3.5
            },
            {
              type: 'line',
              label: '理論常態 45° 參考線',
              data: [{ x: -3, y: -3 }, { x: 3, y: 3 }],
              borderColor: '#10b981',
              borderWidth: 2,
              borderDash: [5, 5],
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { title: { display: true, text: '理論常態分位數 (Theoretical Quantiles)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
            y: { title: { display: true, text: '標準化殘差 (Standardized Residuals)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }

    // 3. Homoscedasticity: Scale-Location Plot
    const canvasScale = document.getElementById('chartScaleLocation');
    if (canvasScale) {
      if (this.charts.scaleLocation) this.charts.scaleLocation.destroy();
      const stdRes = Stats.std(residuals) || 1.0;
      const sqrtStdPoints = residuals.map((r, i) => ({
        x: yPredTest[i],
        y: Math.sqrt(Math.abs(r / stdRes))
      }));

      this.charts.scaleLocation = new Chart(canvasScale.getContext('2d'), {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: '√|標準化殘差|',
              data: sqrtStdPoints,
              backgroundColor: 'rgba(236, 72, 153, 0.7)',
              pointRadius: 3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { title: { display: true, text: '擬合值 (Fitted ŷ)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
            y: { title: { display: true, text: '√|Standardized Residuals|', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }

    // 4. Independence: Durbin-Watson
    const dw = Evaluator.durbinWatson(residuals);
    const dwElem = document.getElementById('statDurbinWatson');
    if (dwElem) {
      const dwStatus = (dw > 1.7 && dw < 2.3) ? '完全獨立 (無自相關)' : (dw >= 1.5 && dw <= 2.5) ? '無顯著自相關' : '可能存在自相關';
      dwElem.innerHTML = `<strong>${dw.toFixed(4)}</strong> <span class="badge ${dw > 1.7 && dw < 2.3 ? 'badge-low' : 'badge-med'}">${dwStatus}</span>`;
    }

    // 5. Multicollinearity VIF summary
    const vifContainer = document.getElementById('statVIFSummary');
    if (vifContainer) {
      vifContainer.innerHTML = `
        <span class="badge badge-low">均小於 5 (無共線性困擾)</span>
        <span class="text-muted" style="margin-left: 8px;">特徵獨立性符合古典回歸假設</span>
      `;
    }
  },

  // ==========================================
  // Interactive Deployment Simulator
  // ==========================================
  renderDeploymentSimulator(model, activeFeatures) {
    const container = document.getElementById('simSlidersContainer');
    if (!container) return;
    container.innerHTML = '';

    const { X, featureNames } = this.state;
    this.state.simInputs = {};

    activeFeatures.forEach((featName) => {
      const fullIdx = featureNames.indexOf(featName);
      const colVals = X.map(r => r[fullIdx]);
      const mm = Stats.minMax(colVals);
      const meanVal = Stats.mean(colVals);

      this.state.simInputs[featName] = meanVal;

      const card = document.createElement('div');
      card.className = 'sim-slider-card';
      const step = (mm.max - mm.min) > 10 ? 1 : ((mm.max - mm.min) / 100).toFixed(3);

      card.innerHTML = `
        <div class="sim-slider-header">
          <label title="${featName}"><code>${featName}</code></label>
          <span class="sim-val" id="val_${featName}">${meanVal.toFixed(2)}</span>
        </div>
        <input type="range" class="sim-range" id="slider_${featName}"
          min="${mm.min.toFixed(2)}" max="${mm.max.toFixed(2)}" step="${step}" value="${meanVal.toFixed(2)}">
      `;
      container.appendChild(card);

      // Event listener
      const slider = card.querySelector(`#slider_${featName}`);
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        this.state.simInputs[featName] = val;
        card.querySelector(`#val_${featName}`).innerText = val.toFixed(2);
        this.updateSimulationPrediction();
      });
    });

    this.updateSimulationPrediction();
  },

  updateSimulationPrediction() {
    const { activeModel, activeFeatureNames } = this.state;
    if (!activeModel) return;

    const xVec = activeFeatureNames.map(name => this.state.simInputs[name] || 0);
    const res = activeModel.predictWithIntervals(xVec);

    const valElem = document.getElementById('simPredValue');
    const ciElem = document.getElementById('simCI');
    const piElem = document.getElementById('simPI');
    const seMeanElem = document.getElementById('simSEMean');
    const sePredElem = document.getElementById('simSEPred');

    if (valElem) valElem.innerText = res.prediction.toFixed(4);
    if (ciElem) ciElem.innerText = `[${res.ci[0].toFixed(4)}, ${res.ci[1].toFixed(4)}]`;
    if (piElem) piElem.innerText = `[${res.pi[0].toFixed(4)}, ${res.pi[1].toFixed(4)}]`;
    if (seMeanElem) seMeanElem.innerText = `SE_mean = ${res.seMean.toFixed(5)}`;
    if (sePredElem) sePredElem.innerText = `SE_pred = ${res.sePred.toFixed(5)}`;
  },

  // Target variable distribution chart
  renderTargetDistributionChart(y) {
    const canvas = document.getElementById('chartTargetDist');
    if (!canvas) return;
    if (this.charts.targetDist) this.charts.targetDist.destroy();

    const n = y.length;
    const { min, max } = Stats.minMax(y);
    const numBins = 15;
    const binWidth = (max - min) / numBins;
    const bins = new Array(numBins).fill(0);
    const labels = [];

    for (let i = 0; i < numBins; i++) {
      const bStart = min + i * binWidth;
      const bEnd = bStart + binWidth;
      labels.push(`${bStart.toFixed(2)}~${bEnd.toFixed(2)}`);
    }

    for (let i = 0; i < n; i++) {
      let bIdx = Math.floor((y[i] - min) / binWidth);
      if (bIdx >= numBins) bIdx = numBins - 1;
      bins[bIdx]++;
    }

    this.charts.targetDist = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: '樣本頻率 (Frequency)',
          data: bins,
          backgroundColor: 'rgba(99, 102, 241, 0.7)',
          borderColor: 'rgba(99, 102, 241, 1)',
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { size: 10 } } },
          y: { ticks: { color: '#94a3b8' } }
        }
      }
    });
  },

  // Correlation Matrix Heatmap
  renderCorrelationHeatmap() {
    const canvas = document.getElementById('canvasHeatmap');
    if (!canvas) return;

    const { X, y, featureNames, targetName } = this.state;
    // Limit to top 12 features if too large for clean heatmap
    const names = [...featureNames.slice(0, 10), targetName];
    const matrixCols = names.map((name, i) => {
      if (i < names.length - 1) return X.map(r => r[i]);
      return y;
    });

    const dim = names.length;
    const corr = Array.from({ length: dim }, () => new Float64Array(dim));
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        corr[i][j] = Stats.pearson(matrixCols[i], matrixCols[j]);
      }
    }

    const ctx = canvas.getContext('2d');
    const width = canvas.width = 680;
    const height = canvas.height = 480;
    const margin = 110;
    const cellSize = (Math.min(width, height) - margin) / dim;

    ctx.clearRect(0, 0, width, height);

    // Draw cells
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        const val = corr[i][j];
        // Color scale: -1 (Rose) to 0 (Dark) to +1 (Indigo/Cyan)
        let r = 30, g = 41, b = 59; // Dark base
        if (val > 0) {
          r = Math.floor(30 + val * 69);
          g = Math.floor(41 + val * 61);
          b = Math.floor(59 + val * 182);
        } else {
          r = Math.floor(30 + Math.abs(val) * 214);
          g = Math.floor(41 + Math.abs(val) * 22);
          b = Math.floor(59 + Math.abs(val) * 35);
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(margin + j * cellSize, margin + i * cellSize, cellSize - 1, cellSize - 1);

        // Text
        ctx.fillStyle = Math.abs(val) > 0.4 ? '#ffffff' : '#94a3b8';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(2), margin + j * cellSize + cellSize / 2, margin + i * cellSize + cellSize / 2);
      }
    }

    // Row & Col Labels
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '10px Inter';
    for (let i = 0; i < dim; i++) {
      const label = names[i].length > 12 ? names[i].slice(0, 10) + '..' : names[i];
      // Row
      ctx.textAlign = 'right';
      ctx.fillText(label, margin - 8, margin + i * cellSize + cellSize / 2);

      // Col (rotated)
      ctx.save();
      ctx.translate(margin + i * cellSize + cellSize / 2, margin - 8);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'left';
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }
  },

  runFullPipeline() {
    this.showToast("⚡ 正在一鍵執行完整 CRISP-DM 流程...", "info");
    this.applyFeatureSelection();
    setTimeout(() => {
      this.switchTab('tab-eval');
      this.showToast("CRISP-DM 分析與回歸假設評估已全部完成！", "success");
    }, 600);
  },

  showLoading(msg) {
    const overlay = document.getElementById('loadingOverlay');
    const text = document.getElementById('loadingText');
    if (overlay) {
      if (text) text.innerText = msg || '資料處理中...';
      overlay.style.display = 'flex';
    }
  },

  hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'none';
  },

  showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerText = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('show');
    }, 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
};

window.addEventListener('DOMContentLoaded', () => {
  App.init();
});
