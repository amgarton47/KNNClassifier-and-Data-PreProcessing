package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;

import ml.data.DataSet;
import ml.data.Example;

/**
 * A simple Perceptron classification model.
 * 
 * @author Aidan Garton
 *
 */
public class PerceptronClassifier implements Classifier {
	protected int numIterations = 10;
	protected ArrayList<Double> w;
	protected double b = 0;
	protected DataSet data;

	public PerceptronClassifier() {
	}

	@Override
	public void train(DataSet data) {
		this.data = data;
		// initialize weight vectors to all zeroes
		int numFeatures = data.getAllFeatureIndices().size();
		w = new ArrayList<Double>(Collections.nCopies(numFeatures, 0.0));

		for (int i = 0; i < numIterations; i++) {
			// randomly shuffle the data
			Collections.shuffle(data.getData());

			for (Example example : data.getData()) {
				double prediction = getPrediction(example);
				double y = example.getLabel();

				if (prediction * y <= 0) { // if we mis-classify example based on weights w
					for (int j = 0; j < w.size(); j++) {
						w.set(j, w.get(j) + y * example.getFeature(j));
					}

					b += y;
				}
			}
		}
	}

	/**
	 * helper function for getting prediction of a data example
	 * 
	 * @param example - the example to predict a label for
	 * @return - the predicted "point in space" of the example
	 */
	protected double getPrediction(Example example) {
		double prediction = b;
		for (int i = 0; i < w.size(); i++) {
			prediction += w.get(i) * example.getFeature(i);
		}
		return prediction;
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example) >= 0 ? 1.0 : -1.0;
	}

	/**
	 * setter method for the number of training Iterations
	 * 
	 * @param numIterations - the number of iterations to train our model on
	 */
	public void setIterations(int numIterations) {
		this.numIterations = numIterations;
	}

	@Override
	public String toString() {
		String str = "";

		for (int i = 0; i < data.getAllFeatureIndices().size(); i++) {
			str += data.getAllFeatureIndices().toArray()[i] + ":" + w.get(i) + " ";
		}
		return str + b;
	}
}
