package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;

import ml.data.DataSet;
import ml.data.Example;

/**
 * A perceptron classification model that averages the weights and bias across
 * training.
 * 
 * @author Aidan Garton
 *
 */
public class AveragePerceptronClassifier extends PerceptronClassifier { // <-- extra credit
	private ArrayList<Double> u;
	private double b2 = 0;
	private int updated = 0, total = 0;

	public AveragePerceptronClassifier() {
	}

	public void train(DataSet data) {
		// initialize weight vectors to all zeroes
		this.data = data;
		int numFeatures = data.getAllFeatureIndices().size();
		w = new ArrayList<Double>(Collections.nCopies(numFeatures, 0.0));
		u = new ArrayList<Double>(Collections.nCopies(numFeatures, 0.0));

		for (int i = 0; i < numIterations; i++) {
			// randomly shuffle the data
			Collections.shuffle(data.getData());
			for (Example example : data.getData()) {
				double prediction = super.getPrediction(example);

				if (prediction * example.getLabel() <= 0) { // if we mis-classify example based on weights w
					for (int j = 0; j < u.size(); j++) {
						u.set(j, u.get(j) + updated * w.get(j));
					}

					b2 += b * updated;

					// update all the perceptron weights
					for (int k = 0; k < w.size(); k++) {
						w.set(k, w.get(k) + example.getLabel() * example.getFeature(k));
					}

					b += example.getLabel();
					updated = 0;
				}

				updated++;
				total++;
			}
		}

		// divide all of the aggregate weights by the total number of examples
		for (int i = 0; i < u.size(); i++) {
			u.set(i, u.get(i) / total);
		}
		b2 /= total;

		// for classification purposes
		b = b2;
		w = u;
	}
}